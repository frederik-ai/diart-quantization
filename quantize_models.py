import time
import os
from copy import deepcopy

import pandas as pd
from tqdm import tqdm
from einops import rearrange
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
# diart
import diart
import diart.inference
import diart.sources
from diart.blocks.embedding import OverlappedSpeechPenalty, TemporalFeatureFormatter
# pyannote
from pyannote.audio import Model
from pyannote.database import registry
from pyannote.audio.tasks import Segmentation, SpeakerEmbedding
from pyannote.audio.core.model import Model
# torch
import torch
import torch.ao.quantization
import torch.ao.quantization.quantizer
from torch.ao.quantization.quantize_pt2e import (
   prepare_pt2e, convert_pt2e, prepare_qat_pt2e
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
   XNNPACKQuantizer,
   get_symmetric_quantization_config,
)
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
# speechbrain
from speechbrain.inference.encoders import MelSpectrogramEncoder
from speechbrain.inference.speaker import EncoderClassifier

from settings import (
   audio_sample_rate, ami_calibration_wav_dir, runtime_evaluation_out_path,
   pyannote_segmentation_models, pyannote_embedding_models, speechbrain_embedding_models
)
from utils import load_random_ami_data_samples

# TODO: Also fuse Linear layers and their activations
def fuse_all_conv_and_batchnorm_layers(model):
   """
   Fuse all Conv layers with their corresponding BatchNorm layers in a model with potential submodules.
   The problem with PyTorch module fusion is that it does not work recursively for nested modules.
   Therefore, we need to manually traverse all children modules of the model and then apply module fusion.
   Assumes that all Conv layers contain the string 'conv' in their name and all BatchNorm layers contain the string 'bn' in their name.
   Note: Only Conv->BatchNorm fusion is supported by PyTorch. Not BatchNorm->Conv which some of the diart models use.
   
   Args:
      model (torch.nn.Module): The model to fuse.
      
   Returns:
      torch.nn.Module: The fused model.
   """
   child_modules = dict(model.named_children())
   child_modules_list = list(child_modules.items())
   for child_idx, (module_name, module) in enumerate(child_modules_list):
      # If the module itself is a container for other modules, then recursively apply this function
      if len(list(module.children())) > 0 or isinstance(module, torch.nn.Sequential):
         module = fuse_all_conv_and_batchnorm_layers(module)
         setattr(model, module_name, module)
      else:
         # If current module is a Conv layer and there exists another module after it
         if 'conv' in module_name and child_idx < len(child_modules_list)-1:
            next_module_name, _ = child_modules_list[child_idx+1]
            # Get the BatchNorm layer name; Expected is: conv --> bn, conv1 -> bn1, conv2 -> bn2, ...
            # Not expected is: conv1 -> bn2, ...
            # expected_batchnorm_layer_name = module_name.replace('conv', 'bn')
            # if expected_batchnorm_layer_name == next_module_name:
            #    layers_to_fuse = [[module_name, next_module_name]]
            #    model = torch.quantization.fuse_modules(model, layers_to_fuse)
            if 'bn' in next_module_name:
               layers_to_fuse = [[module_name, next_module_name]]
               model = torch.quantization.fuse_modules(model, layers_to_fuse)
   return model

def preprocess_inputs_for_embedding_model(waveform_segments, segmentation_output):
   """
   Preprocesses the input data for the embedding model such that it has the exact dimensions the model will encounter in the diart pipeline.
   This is necessary for static quantization because the model loses its ability to adapt to different input shapes.
   
   Args:
      waveform_segments (torch.Tensor): The input waveform segments.
      segmentation_output (torch.Tensor): The output of the segmentation model for the waveform_segments.
      
   Returns:
      tuple: The preprocessed waveform inputs and segmentation weights for the embedding model.
   """
   # From diart.blocks.embedding.OverlapAwareSpeakerEmbedding
   # This code applies a penalty on overlapping speech and low-confidence regions to speaker segmentation scores.
   # Purpose is that embedding model places less focus on such low-confidence regions.
   # For more information refer to the diart paper: (https://github.com/juanmc2005/diart/blob/main/paper.pdf)
   overlapped_speech_penalty = OverlappedSpeechPenalty()
   segmentation_output = overlapped_speech_penalty(segmentation_output)
   
   # Preprocess input according to diart.blocks.embedding.SpeakerEmbedding.__call__
   # Purpose is to extract the number of speakers from the segmentation output and reshape weights as well as model inputs accordingly.
   # Embedding model will receive a batch of inputs and weights, where the batch dimension is multiplied by the number of speakers.
   waveform_formatter = TemporalFeatureFormatter()
   weights_formatter = TemporalFeatureFormatter()
   inputs = waveform_segments
   weights = segmentation_output
   inputs = waveform_formatter.cast(inputs)
   # This line is unnecessary, because in our case the shape already is (batch, channel, sample)
   # inputs = rearrange(inputs, "batch sample channel -> batch channel sample")
   weights = weights_formatter.cast(weights)
   batch_size, _, num_speakers = weights.shape
   inputs = inputs.repeat(1, num_speakers, 1)
   weights = rearrange(segmentation_output, "batch frame spk -> (batch spk) frame")
   inputs = rearrange(inputs, "batch spk sample -> (batch spk) 1 sample")
   return inputs, weights

def quantize_static_pt2e_with_calibration(segmentation_model, embedding_model, calibration_data, batch_size=1, model_to_quantize="segmentation",
                                          compile_quantized_models=False, quantization_backend="x86"):
   """
   Statically quantize a model such that it can be used within the diart framework. Uses the PyTorch 2 Export Quantization (pt2e) API.
   The model will be quantized to int8 precision. You can choose to quantize only the segmentation model, only the embedding model or both.
   You also have the choice of using the x86 or xnnpack quantization backend. Use x86 for server inference and xnnpack for mobile inference.
   In case of xnnpack, the model will have to be lowered to the target device. For more information refer to (https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html).
   
   Args:
      segmentation_model (torch.nn.Module): The segmentation model to quantize.
      embedding_model (torch.nn.Module): The embedding model to quantize.
      calibration_data (torch.Tensor): The calibration data to use for quantization.
      batch_size (int): The batch size to use for calibration.
      model_to_quantize (str): The model to quantize. Must be one of ['segmentation', 'embedding', 'both'].
      compile_quantized_models (bool): Whether to compile the quantized models for faster inference. Compilation takes place during the first forward pass.
      quantization_backend (str): The quantization backend to use. Must be one of ['x86', 'xnnpack'].
      
   Returns:
      The quantized model.
   """
   
   if model_to_quantize not in ["segmentation", "embedding", "both"]:
      raise ValueError(f"model_to_quantize must be one of ['segmentation', 'embedding', 'both'] but got {model_to_quantize}.")
   
   quantizer = None
   if quantization_backend == "x86":
      quantizer = X86InductorQuantizer().set_global(xiq.get_default_x86_inductor_quantization_config(is_dynamic=False))
      # With this, the model will be compiled to a C++ wrapper instead of the default Python wrapper
      # Only has an effect if compile_quantized_models is True
      import torch._inductor.config as config
      config.cpp_wrapper = True
   elif quantization_backend == "xnnpack":
      quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
   else:
      raise ValueError(f"quantization_backend must be one of ['x86', 'xnnpack'] but got {quantization_backend}.")
   
   segmentation_model.eval()
   embedding_model.eval()
   
   example_input = calibration_data[:batch_size]
   if batch_size == 1 and example_input.dim() == 2:
      example_input = example_input.unsqueeze(0)
   
   # Prepare models for static quantization
   if model_to_quantize == "segmentation" or model_to_quantize == "both":
      segmentation_model = torch.export.export_for_training(segmentation_model, (example_input,)).module()
      segmentation_model = prepare_pt2e(segmentation_model, quantizer)
      torch.ao.quantization.allow_exported_model_train_eval(segmentation_model) # for compatibility with diart code
   if model_to_quantize == "embedding" or model_to_quantize == "both":
      segmentation_output = segmentation_model(example_input)
      inputs, weights = preprocess_inputs_for_embedding_model(example_input, segmentation_output)
      embedding_model = torch.export.export_for_training(embedding_model, (inputs, weights)).module()
      embedding_model = prepare_pt2e(embedding_model, quantizer)
      torch.ao.quantization.allow_exported_model_train_eval(embedding_model)
   
   # Calibrate the models
   from diart import SpeakerDiarization, SpeakerDiarizationConfig
   AMI_DIART_HYPERPARAMS = {
        "step": 0.5,
        "latency": 0.5,
        "tau_active": 0.507,
        "rho_update": 0.006,
        "delta_new": 1.057
    }
   config = SpeakerDiarizationConfig(
      segmentation=segmentation_model,
      embedding=embedding_model,
      device=torch.device("cpu"),
      **AMI_DIART_HYPERPARAMS
   )
   diarization_pipeline = SpeakerDiarization(config)
   for calibration_wav in os.listdir(ami_calibration_wav_dir):
      full_calibration_data_path = os.path.join(ami_calibration_wav_dir, calibration_wav)
      audio_source = diart.sources.FileAudioSource(full_calibration_data_path, sample_rate=audio_sample_rate)
      streaming_inference = diart.inference.StreamingInference(diarization_pipeline, audio_source, batch_size=batch_size)
      _ = streaming_inference()

   # Convert to int8
   segmentation_model_int8 = None
   embedding_model_int8 = None
   if model_to_quantize == "segmentation" or model_to_quantize == "both":
      segmentation_model_int8 = convert_pt2e(segmentation_model)
      torch.ao.quantization.move_exported_model_to_eval(segmentation_model_int8)
      torch.ao.quantization.allow_exported_model_train_eval(segmentation_model_int8)
   if model_to_quantize == "embedding" or model_to_quantize == "both":
      embedding_model_int8 = convert_pt2e(embedding_model)
      torch.ao.quantization.move_exported_model_to_eval(embedding_model_int8)
      torch.ao.quantization.allow_exported_model_train_eval(embedding_model_int8)
   with torch.no_grad():
      segmentation_model_int8 = torch.compile(segmentation_model_int8) if compile_quantized_models else segmentation_model_int8
      embedding_model_int8 = torch.compile(embedding_model_int8) if compile_quantized_models else embedding_model_int8
   
   if model_to_quantize == "segmentation":
      return segmentation_model_int8
   elif model_to_quantize == "embedding":
      return embedding_model_int8
   else: # both
      return segmentation_model_int8, embedding_model_int8
   
class QATLightningModule(Model):
   def __init__(self, model, lr=1e-6):
      super().__init__()
      self.model = model
      self.task = model.task
      self.lr = lr
      
   def forward(self, x):
      pred = self.model(x)
      return pred

   # Overwrite this method from the Model class to allow for a custom learning rate
   def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
   
def pt2e_quantization_aware_training(segmentation_model, embedding_model, calibration_data, max_epochs=20, batch_size=1, early_stopping=True, model_to_quantize="segmentation",
                                     compile_quantized_models=False, quantization_backend="x86"):
   """
   Do quantization aware training (QAT) using the PyTorch 2 Export Quantization (pt2e) API. Currently only QAT of the segmentation model is supported.
   You have the choice of using the x86 or xnnpack quantization backend. Use x86 for server inference and xnnpack for mobile inference.
   In case of xnnpack, the model will have to be lowered to the target device. For more information refer to (https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html).
   
   Args:
      segmentation_model (torch.nn.Module): The segmentation model to quantize.
      embedding_model (torch.nn.Module): The embedding model to quantize.
      calibration_data (torch.Tensor): The calibration data to use for quantization.
      max_epochs (int): The maximum number of epochs to train the model.
      batch_size (int): The batch size to use for calibration and training.
      early_stopping (bool): Whether to use early stopping during training.
      model_to_quantize (str): The model to quantize. Must be one of ['segmentation', 'embedding', 'both']. Currenntly only 'segmentation' is supported.
      
   Returns:
      The quantized model.
   """
   
   if model_to_quantize in ["embedding", "both"]:
      raise ValueError('Quantization Aware Training of the embedding model is not yet supported.')
   elif not model_to_quantize == "segmentation":
      raise ValueError(f"model_to_quantize must be one of ['segmentation', 'embedding', 'both'] but got {model_to_quantize}.")
   
   quantizer = None
   if quantization_backend == "x86":
      quantizer = X86InductorQuantizer().set_global(xiq.get_default_x86_inductor_quantization_config(is_dynamic=False))
      # With this, the model will be compiled to a C++ wrapper instead of the default Python wrapper
      # Only has an effect if compile_quantized_models is True
      import torch._inductor.config as config
      config.cpp_wrapper = True
   elif quantization_backend == "xnnpack":
      quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
   else:
      raise ValueError(f"quantization_backend must be one of ['x86', 'xnnpack'] but got {quantization_backend}.")
   
   receptive_field = None
   num_frames = None
   if model_to_quantize == "segmentation":
      receptive_field = segmentation_model.receptive_field
      num_frames = segmentation_model.num_frames
      
   segmentation_model.eval()
   embedding_model.eval()
   
   example_input = calibration_data[:batch_size]
   if batch_size == 1 and example_input.dim() == 2:
      example_input = example_input.unsqueeze(0)
      
   if model_to_quantize == "segmentation":
      segmentation_model = torch.export.export_for_training(segmentation_model, (example_input,)).module()
      segmentation_model = prepare_qat_pt2e(segmentation_model, quantizer)
      torch.ao.quantization.allow_exported_model_train_eval(segmentation_model)

   # Train
   registry.load_database("AMI-diarization-setup/pyannote/database.yml")
   ami = registry.get_protocol("AMI.SpeakerDiarization.mini")

   segmentation_model.task = Segmentation(ami, duration=5, batch_size=batch_size)
   embedding_model.task = SpeakerEmbedding(ami, duration=5)
   embedding_model.task.batch_size = batch_size
   
   trainer_callbacks = [EarlyStopping(monitor="DiarizationErrorRate", patience=3)] if early_stopping else None
   # run training on CPU because with cuda LSTM fake quantization throws errors
   trainer = Trainer(max_epochs=max_epochs, callbacks=trainer_callbacks, accelerator="cpu", limit_train_batches=100)
   # We have to wrap the model in a LightningModule to use PyTorch Lightning
   qat_model = None
   if model_to_quantize == "segmentation":
      qat_model = QATLightningModule(segmentation_model)
   qat_model.receptive_field = receptive_field
   qat_model.num_frames = num_frames
   trainer.fit(qat_model)
   
   trained_qat_model = qat_model.model
   trained_qat_model = convert_pt2e(trained_qat_model)
   torch.ao.quantization.move_exported_model_to_eval(trained_qat_model)
   torch.ao.quantization.allow_exported_model_train_eval(trained_qat_model)
   
   with torch.no_grad():
      trained_qat_model = torch.compile(trained_qat_model) if compile_quantized_models else trained_qat_model
   
   return trained_qat_model
   

def calculate_model_runtime(model, waveform_segments, num_decimals=2, add_batch_dimension=False, custom_inference_fn=None, weights=None):
   """
   Evaluate the average runtime of a model on a given set of waveform segments. For a single pass a batch size of 1 is used.
   
   Args:
      model (torch.nn.Module): The model to evaluate.
      waveform_segments (torch.Tensor): The waveform segments to evaluate the model on.
      num_decimals (int): The number of decimals to round the average runtime to.
      add_batch_dimension (bool): Whether to add a batch dimension to the waveform segments. Otherwise the input will have no batch dimension.
      custom_inference_fn (callable): A custom inference function to use instead of the model's __call__ method.
      weights (torch.Tensor): The weights to use for the embedding model. Only necessary if the model is a statically quantized embedding model.
   
   Returns:
      float: The average runtime of the model in milliseconds.
   """
   
   def run_inference(model, model_input, weights=None):
      if weights is None:
         model_input = (model_input,)
      else:
         model_input, weights = preprocess_inputs_for_embedding_model(model_input, weights)
         # only take first element in (batch+speaker) dimension. Otherwise unfair comparison to non-weighted inference with batch size 1
         model_input, weights = model_input[:1], weights[:1]
         model_input = (model_input, weights)
      
      if custom_inference_fn:
         _ = custom_inference_fn(*model_input)
      else:
         _ = model(*model_input)
   
   # do inference to warm up the model -> construction of computational graph for potential model compilation
   waveform_segment = waveform_segments[0].unsqueeze(0) if add_batch_dimension else waveform_segments[0]
   run_inference(model, waveform_segment, weights)
   waveform_segment = waveform_segments[1].unsqueeze(0) if add_batch_dimension else waveform_segments[1]
   run_inference(model, waveform_segment, weights)
   
   inference_times = torch.zeros(len(waveform_segments))
   for i, waveform_segment in enumerate(tqdm(waveform_segments)):
      waveform_segment = waveform_segment.unsqueeze(0) if add_batch_dimension else waveform_segment
      start_time = time.time()
      run_inference(model, waveform_segment, weights)
      inference_times[i] = time.time() - start_time

   average_runtime_in_seconds = torch.mean(inference_times).item()
   average_runtime_in_milliseconds = average_runtime_in_seconds * 1000
   return round(average_runtime_in_milliseconds, num_decimals)

def evaluate_optimized_model_runtime(segmentation_model, embedding_model, waveform_segments, do_static_quantization, 
                                     model_to_evaluate="segmentation", **kwargs):
   """
   Evaluate the runtime of a model on a given set of waveform segments. The model will be optimized using different types of dynamic quantization as well as static quantization.
   Regarding dynamic quantization, this model returns the quantized model with the lowest runtime. Because it is necessary for stati quantization, you have to provide both a
   segmentation and an embedding model. You specifiy with the model_to_evaluate parameter which of the two models you actually want to quantize.
   
   Args:
      segmentation_model (torch.nn.Module): The segmentation model to evaluate.
      embedding_model (torch.nn.Module): The embedding model to evaluate.
      waveform_segments (torch.Tensor): The waveform segments to evaluate the model on.
      do_static_quantization (bool): Whether to do static quantization.
      model_to_evaluate (str): The model to evaluate. Must be one of ['segmentation', 'embedding'].
      **kwargs: Additional keyword arguments for the inference of the model.
      
   Returns:
      tuple: Firstly, a dictionary containing the runtime of the baseline model, all the dynamically quantized models and the statically quantized model.
      Secondly the dynamically quantized model with the lowest runtime. Lastly, the statically quantized model.
   """
   
   model = None
   if model_to_evaluate == "segmentation":
      model = segmentation_model
   elif model_to_evaluate == "embedding":
      model = embedding_model
   else:
      raise ValueError(f"model_to_evaluate must be one of ['segmentation', 'embedding'] but got {model_to_evaluate}.")
   
   dynamically_quantized_models = []
   dynamically_quantized_model_runtimes = [] 
   
   results = {}
   model.eval()
   model = model.to("cpu")
   waveform_segments = waveform_segments.to("cpu")
   
   baseline_runtime = calculate_model_runtime(model, waveform_segments, **kwargs)
   results["baseline"] = baseline_runtime
   
   model_fused = deepcopy(model)
   model_fused.eval()
   model_fused = fuse_all_conv_and_batchnorm_layers(model_fused)
   
   # (name_of_quantization, layers_to_quantize)
   dynamic_quantization_combinations = [
      ("ptdq_int8", {}),
      ("ptdq_linear_int8", {torch.nn.Linear}),
      ("ptdq_conv1d_int8", {torch.nn.Conv1d}),
      ("ptdq_conv2d_int8", {torch.nn.Conv2d}),
      ("ptdq_conv_int8", {torch.nn.Conv1d, torch.nn.Conv2d}),
      ("ptdq_linear_conv_int8", {torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear}),
      ("ptdq_lstm_int8", {torch.nn.LSTM}),
      ("ptdq_conv_lstm_linear_int8", {torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.LSTM, torch.nn.Linear})
   ]
   
   for quantization_name, layers_to_quantize in dynamic_quantization_combinations:
      model_ptdq_int8 = torch.ao.quantization.quantize_dynamic(model_fused, layers_to_quantize, dtype=torch.qint8)
      results[quantization_name] = calculate_model_runtime(model_ptdq_int8, waveform_segments, **kwargs)
      dynamically_quantized_models.append(model_ptdq_int8)
      dynamically_quantized_model_runtimes.append(results[quantization_name])
   fastest_dynamically_quantized_model_index = torch.argmin(torch.tensor(dynamically_quantized_model_runtimes)).item()
   fastest_dynamically_quantized_model = dynamically_quantized_models[fastest_dynamically_quantized_model_index]
   
   if do_static_quantization:
      # Compiling the models can crash without errors; Therefore set compile_quantized_models to False
      # Note: Actual compilation process is run during the first inference of the model
      segmentation_model.eval()
      embedding_model.eval()
      model_ptsq_int8 = quantize_static_pt2e_with_calibration(segmentation_model, embedding_model, waveform_segments, model_to_quantize=model_to_evaluate, batch_size=1,
                                                               compile_quantized_models=False)
      if model_to_evaluate == "segmentation":
         results["ptsq_int8"] = calculate_model_runtime(model_ptsq_int8, waveform_segments, **kwargs)
      else:
      # results["ptsq__pt2e_int8"] = calculate_model_runtime(model_ptsq_int8, waveform_segments, custom_inference_fn=kwargs.get("custom_inference_fn", None))
         dummy_weights = segmentation_model(waveform_segments[0])
         dummy_weights = dummy_weights.squeeze(0)
         results["ptsq_int8"] = calculate_model_runtime(model_ptsq_int8, waveform_segments, weights=dummy_weights, **kwargs)
   else:
      model_ptsq_int8 = None
      results["ptsq_int8"] = None
   
   return results, fastest_dynamically_quantized_model, model_ptsq_int8


if __name__ == '__main__':
   test_waveform_segments = load_random_ami_data_samples(split="test", num_samples=48, audio_duration=5)
   train_waveform_segments = load_random_ami_data_samples(split="train", num_samples=48, audio_duration=5)
   
   pyannote_models = ["pyannote/embedding", "pyannote/segmentation", "pyannote/segmentation-3.0", "pyannote/wespeaker-voxceleb-resnet34-LM"]
   speechbrain_models = ["speechbrain/spkrec-xvect-voxceleb", "speechbrain/spkrec-ecapa-voxceleb", "speechbrain/spkrec-ecapa-voxceleb-mel-spec",
                        "speechbrain/spkrec-resnet-voxceleb"]

   total_results = []   
   if os.path.exists(runtime_evaluation_out_path):
      df_existing = pd.read_csv(runtime_evaluation_out_path)
      total_results = df_existing.to_dict(orient="records")
   already_evaluated_models = {result["model"] for result in total_results}

   model_sizes = []
   os.makedirs("quantized_models", exist_ok=True)
   
   default_segmentation_model = pyannote_segmentation_models[0]
   default_embedding_model = pyannote_embedding_models[0]
   for model_name in pyannote_segmentation_models+pyannote_embedding_models+speechbrain_embedding_models:
      print("Model:", model_name)
      
      # skip model if it already exists in the data
      if model_name in already_evaluated_models:
         print(f"Model {model_name} already exists in {runtime_evaluation_out_path}. Skipping this model.")
         continue
      
      segmentation_model = None
      embedding_model = None
      embedding_model_inference_fn = None # speechbrain models require a custom inference function
      model_to_evaluate = None
      add_batch_dimension = False  # necessary for pyannote models; does not work for speechbrain models
      do_static_quantization = True
      # models where static quantization fails
      if model_name in ["pyannote/wespeaker-voxceleb-resnet34-LM"]+speechbrain_embedding_models:
         do_static_quantization = False
      
      if model_name in pyannote_segmentation_models:
         segmentation_model = Model.from_pretrained(model_name)
         model_to_evaluate = "segmentation"
         add_batch_dimension = True
      elif model_name in pyannote_embedding_models:
         embedding_model = Model.from_pretrained(model_name)
         model_to_evaluate = "embedding"
         add_batch_dimension = True
      elif model_name in speechbrain_embedding_models:
         if "-mel-" in model_name:
            embedding_model = MelSpectrogramEncoder.from_hparams(source=model_name)
            embedding_model_inference_fn = embedding_model.encode_waveform
         else:
            embedding_model = EncoderClassifier.from_hparams(source=model_name)
            embedding_model_inference_fn = embedding_model.encode_batch
         model_to_evaluate = "embedding"
         add_batch_dimension = False
      else:
         raise ValueError(f"Unsupported model: {model_name}")
      
      if not segmentation_model:
         segmentation_model = Model.from_pretrained(default_segmentation_model)
      if not embedding_model:
         embedding_model = Model.from_pretrained(default_embedding_model)
      
      results, ptdq_model, ptsq_model = evaluate_optimized_model_runtime(
         segmentation_model, embedding_model, test_waveform_segments, do_static_quantization, model_to_evaluate=model_to_evaluate,
         add_batch_dimension=add_batch_dimension, custom_inference_fn=embedding_model_inference_fn
      )
      
      results["model"] = model_name
      total_results.append(results)
      print(f"{model_name} - done")
      
      # Save current total results to csv; Do this after each model to avoid losing data in case of a crash
      df = pd.DataFrame(total_results)
      df.set_index("model", inplace=True)
      df.to_csv(runtime_evaluation_out_path)
      print(f"Results saved to {runtime_evaluation_out_path}")
      
      # Save dynamically quantized model to .pt
      model_name_without_slash = model_name.replace("/", "_")
      torch.save(ptdq_model, f"quantized_models/{model_name_without_slash}.pt")
      ptdq_model.eval()
      
      # Save ptsq model
      # example_input = test_waveform_segments[0].unsqueeze(0)
      # if model_to_evaluate == "embedding":
      #    segmentation_output = segmentation_model(example_input)
      #    ptsq_model_ep = torch.export.export(embedding_model, (example_input, segmentation_output))
      # else:
      #    ptsq_model_ep = torch.export.export(segmentation_model, (example_input,))
      #ptsq_model_ep = torch.export.export(ptsq_model, (test_waveform_segments[0],))
      # module = ptsq_model_ep.module()
      # ptsq_model_path = f"quantized_models/{model_name_without_slash}_ptsq.pt"
      # torch.export.save(ptsq_model_ep, ptsq_model_path)
      # # Test if quantized model can be loaded and inference can be performed
      # def test_pt2e_model_inference(model_path, waveform_segments):
      #    ep = torch.export.load(model_path)
      #    model = ep.module()
      #    # model.eval() --> message: "Calling eval() is not supported yet." from PyTorch
      #    for waveform_segment in waveform_segments:
      #       _ = model(waveform_segment)
      # test_pt2e_model_inference(ptsq_model_path, test_waveform_segments)      
      
      # def get_model_size(model):
      #    """Refer to: https://discuss.pytorch.org/t/finding-model-size/130275"""
         
      #    param_size = 0
      #    for param in model.parameters():
      #       param_size += param.nelement() * param.element_size()
      #    buffer_size = 0
      #    for buffer in model.buffers():
      #       buffer_size += buffer.nelement() * buffer.element_size()

      #    size_all_mb = (param_size + buffer_size) / 1024**2
      #    return round(size_all_mb, 2)
      
      # model_size_results = {}
      # model_size_results["model"] = model_name
      # model_size_results["baseline"] = get_model_size(model)
      # model_size_results["ptdq"] = get_model_size(quantized_model)
      # model_size_results["ptsq"] = get_model_size(ptsq_model)
      # model_sizes.append(model_size_results)
      
      # print("Baseline model size:", model_size_results["baseline"])
      # print("PTDQ model size:", model_size_results["ptdq"])
      # print("PTSQ model size:", model_size_results["ptsq"])
   
   # # Save model sizes
   # df = pd.DataFrame(model_sizes)
   # df.set_index("model", inplace=True)
   # output_file = "model_sizes.csv"
   # df.to_csv(output_file)
   
   # print(f"Model sizes saved to {output_file}")