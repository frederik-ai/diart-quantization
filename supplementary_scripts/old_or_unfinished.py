# Contains the old or unfinished code that I have written for this project.
# This code is not used in the final version of the project.
# It is just there for documentation or backup purposes.

import torch
from einops import rearrange
from pyannote.audio import Model
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
   XNNPACKQuantizer,
   get_symmetric_quantization_config,
)
from pyannote.database import registry
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from pyannote.audio.tasks import Segmentation, SpeakerEmbedding
from copy import deepcopy
from enum import Enum
from torch.quantization import QuantStub, DeQuantStub

# add quantization stubs
class ModuleWithQuantStubs(torch.nn.Module):
   def __init__(self, module, qconfig, quant_before_dequant=False):
      super().__init__()
      self.quant = torch.quantization.QuantStub(qconfig)
      self.module = module
      self.dequant = torch.quantization.DeQuantStub(qconfig)
      self.quant_before_dequant = quant_before_dequant
   def forward(self, x):
      if self.quant_before_dequant:
         print(x)
         x = self.quant(x)
         print(x)
         x = self.module(x)
         x = self.dequant(x)
      else:
         x = self.dequant(x)
         x = self.module(x)
         x = self.quant(x)
      
      return x
   
class BidirectionalQuantWrapper(torch.quantization.QuantWrapper):
   """
   Essentially the same as torch.quantization.QuantWrapper.
   The limitation of QuantWrapper is it only works in the way fp32 -> quantize -> int8 -> module -> int8 -> dequantize -> fp32.
   This class allows for the following: int8 -> dequantize -> fp32 -> model -> fp32 -> quantize -> int8.
   """
   def __init__(self, module, quant_before_dequant=True):
      super().__init__(module)
      self.quant_before_dequant = quant_before_dequant
      
   def forward(self, x):
      if self.quant_before_dequant:
         # use this to wrap around the whole model
         x = self.quant(x)
         x = self.module(x)
         x = self.dequant(x)
      else:
         # use this to wrap around unsupported modules
         x = self.dequant(x)
         x = self.module(x)
         x = self.quant(x)
      
      return x
   
def get_model_with_quant_stubs(model):
   # https://pytorch.org/blog/quantization-in-practice/
   model_with_quant_stubs = torch.nn.Sequential(torch.quantization.QuantStub(), 
                                                model, 
                                                torch.quantization.DeQuantStub())
   return model_with_quant_stubs

# def quantize_static(model, calibration_data, unsupported_modules_list, quantizazion_backend="x86"):
   
#    # # add quantization stubs
#    # class ModuleWithQuantStubs(torch.nn.Module):
#    #    def __init__(self, module, quant_before_dequant=False):
#    #       super().__init__()
#    #       self.quant = torch.quantization.QuantStub()
#    #       self.module = module
#    #       self.dequant = torch.quantization.DeQuantStub()
#    #       self.quant_before_dequant = quant_before_dequant
#    #    def forward(self, x):
#    #       if self.quant_before_dequant:
#    #          x = self.quant(x)
#    #          x = self.module(x)
#    #          x = self.dequant(x)
#    #       else:
#    #          x = self.dequant(x)
#    #          x = self.module(x)
#    #          x = self.quant(x)
         
#    #       return x
      
#    # model_fp32 = ModuleWithQuantStubs(model, quant_before_dequant=True)
#    model_fp32 = BidirectionalQuantWrapper(model, quant_before_dequant=True)
#    model_fp32.eval()
   
#    for name, module in model_fp32.named_modules():
#      print(name)
#      print(module)
#      print()
      
#    print(model_fp32)   
   
#    # Add quantization stubs around unsupported modules
#    for name, module in model.named_modules():
#       if name in unsupported_modules_list:
#          # setattr(model_fp32, name, BidirectionalQuantWrapper(module, quant_before_dequant=False))
#          model_fp32._modules[name] = BidirectionalQuantWrapper(module, quant_before_dequant=False)
   
#    # for name, module in model.named_modules():
#    #    if name in unsupported_modules_list:
#    #       module.qconfig = None
   
#    # Now print all modules
#    # for name, module in model_fp32.named_modules():
#    #    print(name)
#    #    print(module)
#    #    print()
      
#    print(model_fp32)   
      
#    model_fp32.qconfig = torch.quantization.get_default_qconfig(quantizazion_backend)
#    # TODO: Fuse modules
#    # model_fp32 = torch.quantization.fuse_modules(...)
#    model_fp32 = fuse_all_conv_and_batchnorm_layers(model_fp32)
#    model_fp32 = torch.quantization.prepare(model_fp32)
#    # use calibration data
#    model_fp32(calibration_data)
#    model_int8 = torch.quantization.convert(model_fp32)
   
#    print(model_int8)
   
#    return model_int8

def quantize_static(model, calibration_data, unsupported_modules_list, quantization_backend="x86"):
    model_fp32 = BidirectionalQuantWrapper(model, quant_before_dequant=True)
    model_fp32.eval()
    
    def replace_unsupported_modules(module, unsupported_modules_list):
        for name, child in module.named_children():
            full_name = f"{name}"
            # if full_name in unsupported_modules_list:
            #     print(f"Replacing unsupported module: {full_name}")
            #     module._modules[name] = BidirectionalQuantWrapper(child, quant_before_dequant=False)
            if full_name == "conv1d":
               print(f"Replacing unsupported module: {full_name}")
               module._modules[name] = BidirectionalQuantWrapper(child, quant_before_dequant=False)
            else:
                # Recursively replace in child modules
                replace_unsupported_modules(child, unsupported_modules_list)
    
    replace_unsupported_modules(model_fp32.module, unsupported_modules_list)
    
    with open("model_fp32.txt", "w") as f:
       print(model_fp32, file=f)
    
    model_fp32.qconfig = torch.quantization.get_default_qconfig(quantization_backend)
    # model_fp32 = fuse_all_conv_and_batchnorm_layers(model_fp32)
    model_fp32 = torch.quantization.prepare(model_fp32)
    model_fp32(calibration_data)
    model_int8 = torch.quantization.convert(model_fp32)
    
    with open("model_int8.txt", "w") as f:
         print(model_int8, file=f)
    
    return model_int8
 

def quantize_static_pt2e(model, calibration_data, quantization_backend="x86"):
   from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
   from torch.ao.quantization.quantizer.xnnpack_quantizer import (
      XNNPACKQuantizer,
      get_symmetric_quantization_config,
   )
   model_fp32 = model.eval()
   model_fp32 = torch.export.export_for_training(model, (calibration_data[0].unsqueeze(0),)).module()
   # m = torch.export.export_for_training(model, (calibration_data[:2],), strict=False).module()
   # m = capture_pre_autograd_graph(m, calibration_data)
   quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
   model_fp32 = prepare_pt2e(model_fp32, quantizer)
   for calibration_input in calibration_data:
      # m(calibration_input)
      model_fp32(calibration_input.unsqueeze(0))
   model_int8 = convert_pt2e(model_fp32)
   # Necessary because diart code runs model.eval() after quantization
   torch.ao.quantization.allow_exported_model_train_eval(model_int8)
   return model_int8
      
class QATEmbeddingLightningModule(Model):
   def __init__(self, segmentation_model, embedding_model):
      super().__init__()
      self.embedding_model = embedding_model
      self.segmentation_model = segmentation_model
      self.task = embedding_model.task
      
   def forward(self, x, weights=None):
      segmentation_output = self.segmentation_model(x)
      inputs = rearrange(segmentation_output, "batch sample channel -> batch channel sample")
      batch_size, _, num_speakers = segmentation_output.shape
      inputs = inputs.repeat(1, num_speakers, 1)
      weights = rearrange(segmentation_output, "batch frame spk -> (batch spk) frame")
      inputs = rearrange(inputs, "batch spk sample -> (batch spk) 1 sample")
      # model_pred = self.embedding_model(inputs, weights)
      model_pred = self.embedding_model(x, segmentation_output)
      return model_pred
      # return self.embedding_model(inputs, weights)
      #output = rearrange(
      #   self.embedding_model(x, weights),
      #   "(batch spk) feat -> batch spk feat",
      #   batch=batch_size,
      #   spk=num_speakers,
      #)
      #return output
      #output = rearrange(
      #   self.embedding_model(inputs, weights),
      #   "(batch spk) feat -> batch spk feat",
      #   batch=batch_size,
      #   spk=num_speakers,
      #)
      #return output
      # return self.embedding_model(x, weights)
      
class QATLightningModule(Model):
   def __init__(self, model):
      super().__init__()
      self.model = model
      self.task = model.task
      
   def forward(self, x):
      pred = self.model(x)
      return pred
   
def pt2e_quantization_aware_training(segmentation_model, embedding_model, calibration_data, max_epochs=1, batch_size=1, early_stopping=True, model_to_quantize="segmentation"):
   
   receptive_field = None
   num_frames = None
   if model_to_quantize == "segmentation":
      receptive_field = segmentation_model.receptive_field
      num_frames = segmentation_model.num_frames
   elif model_to_quantize == "embedding":
      receptive_field = embedding_model.receptive_field
      num_frames = embedding_model.num_frames
   elif model_to_quantize == "both":
      raise ValueError('During Quantization Aware Training we can only quantize one model at a time (either "segmentation" or "embedding").')
   else:
      raise ValueError(f"model_to_quantize must be one of ['segmentation', 'embedding'] but got {model_to_quantize}.")
   
   example_input = None
   if batch_size == 1:
      example_input = calibration_data[0].unsqueeze(0)
   else:
      example_input = calibration_data[:batch_size]
   
   # example_input = calibration_data[0].unsqueeze(0)
   # receptive_field = segmentation_model.receptive_field
   # num_frames = segmentation_model.num_frames
   
   def prepare_segmentation_model(segmentation_model):
      segmentation_model = torch.export.export_for_training(segmentation_model, (example_input,)).module()
      quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
      segmentation_model = prepare_qat_pt2e(segmentation_model, quantizer)
      torch.ao.quantization.allow_exported_model_train_eval(segmentation_model)
      return segmentation_model
   
   def prepare_embedding_model(embedding_model):
      segmentation_output = segmentation_model(example_input)
      
      # Preprocess input according to diart.blocks.embedding.SpeakerEmbedding.__call__
      # Necessary because the model needs to know the exact shape of the future input
      # inputs = rearrange(example_input, "batch sample channel -> batch channel sample") # TODO: Uncomment??
      inputs = example_input # TODO REMOVE??
      _, _, num_speakers = segmentation_output.shape
      inputs = inputs.repeat(1, num_speakers, 1)
      weights = rearrange(segmentation_output, "batch frame spk -> (batch spk) frame")
      inputs = rearrange(inputs, "batch spk sample -> (batch spk) 1 sample")
      
      # TODO: Muss das hier nicht "inputs" statt example_input sein??
      embedding_model = torch.export.export_for_training(embedding_model, (example_input, segmentation_output)).module() # TODO: REMOVE
      # embedding_model = torch.export.export_for_training(embedding_model, (inputs, weights)).module()
      # embedding_model = torch.export.export_for_training(embedding_model, (inputs, weights)).module()
      quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
      # embedding_model = prepare_qat_pt2e(embedding_model, quantizer)
      embedding_model = prepare_pt2e(embedding_model, quantizer)
      torch.ao.quantization.allow_exported_model_train_eval(embedding_model)
      return embedding_model
   
   if model_to_quantize == "segmentation":
      segmentation_model = prepare_segmentation_model(segmentation_model)
   elif model_to_quantize == "embedding":
      embedding_model = prepare_embedding_model(embedding_model)
      # embedding_model = prepare_segmentation_model(embedding_model) # TODOD: REMOVE THIS!!
   
   # Train
   registry.load_database("AMI-diarization-setup/pyannote/database.yml")
   ami = registry.get_protocol("AMI.SpeakerDiarization.mini")

   segmentation_model.task = Segmentation(ami, duration=5, batch_size=batch_size)
   # segmentation_model.task.setup()
   
   embedding_model.task = SpeakerEmbedding(ami, duration=5)
   embedding_model.task.batch_size = batch_size
   
   # embedding_model.task.setup()
   early_stopping = False # REMOVE!!
   trainer_callbacks = [EarlyStopping(monitor="DiarizationErrorRate", patience=2)] if early_stopping else None
   # trainer = Trainer(max_epochs=max_epochs, callbacks=trainer_callbacks)
   trainer = Trainer(max_epochs=max_epochs, callbacks=trainer_callbacks, limit_train_batches=0.25) # TODO: Remove limit_train_batches; only for testing to make qat faster
   qat_model = None
   if model_to_quantize == "segmentation":
      qat_model = QATLightningModule(segmentation_model)
   else:
      # diarization_pipeline = SpeakerDiarization(segmentation_model, embedding_model)
      # qat_model = QATLightningModule(diarization_pipeline)
      
      # qat_model = QATLightningModule(embedding_model)
      qat_model = QATEmbeddingLightningModule(segmentation_model, embedding_model)
   qat_model.receptive_field = receptive_field
   qat_model.num_frames = num_frames
   trainer.fit(qat_model)
   
   trained_qat_model = qat_model.model
   trained_qat_model = convert_pt2e(trained_qat_model)
   torch.ao.quantization.move_exported_model_to_eval(trained_qat_model)
   torch.ao.quantization.allow_exported_model_train_eval(trained_qat_model)
   
   # determine diarization error rate on test data
   # diarization_error_rate = pyannote.metrics.diarization.DiarizationErrorRate()
   # for file in tqdm(ami.test()):
   #    reference = file['annotation']
   #    audio_path = file['audio']
      
   #    hypothesis = trained_qat_model(x)
   #    diarization_error_rate(reference, hypothesis)
   #    print()
   # test_diarization = trained_qat_model(test_data)
   # test_metric = test_diarization.suggest_metric()
   # test_metric = test_metric(test_diarization)
   # def test(model, protocol, subset="test"):
   #    from pyannote.audio import Inference
   #    from pyannote.audio.utils.signal import binarize
   #    from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
   #    from pyannote.audio.pipelines.utils import get_devices

   #    (device,) = get_devices(needs=1)
   #    metric = DiscreteDiarizationErrorRate()
   #    files = list(getattr(protocol, subset)())

   #    inference = Inference(QATLightningModule(trained_qat_model), device=device)

   #    for file in files:
   #       reference = file["annotation"]
   #       hypothesis = binarize(inference(file))
   #       uem = file["annotated"]
   #       _ = metric(reference, hypothesis, uem=uem)

   #    return abs(metric)
   # metric_value = test(trained_qat_model, ami)
   # print(f"Diarization Error Rate: {metric_value}")
   
   return trained_qat_model

   # if model_to_quantize == "segmentation":
   #    qat_segmentation_model = QATLightningModule(segmentation_model)
   #    qat_segmentation_model.receptive_field = receptive_field
   #    qat_segmentation_model.num_frames = num_frames
   #    trainer.fit(qat_segmentation_model)
   #    segmentation_model = qat_segmentation_model.model
   # if model_to_quantize == "embedding":
   #    qat_embedding_model = QATLightningModule(embedding_model)
   #    trainer.fit(qat_embedding_model)
   #    embedding_model = qat_embedding_model.model
   
   # # Convert to int8
   # if model_to_quantize == "segmentation":
   #    segmentation_model_int8 = convert_pt2e(segmentation_model)
   #    torch.ao.quantization.move_exported_model_to_eval(segmentation_model_int8) # TODO: also do this for normal static quantization??
   #    torch.ao.quantization.allow_exported_model_train_eval(segmentation_model_int8)
   #    # with torch.no_grad():
   #    #    segmentation_model_int8 = torch.compile(segmentation_model_int8)
   #    return segmentation_model_int8
   # elif model_to_quantize == "embedding":
   #    embedding_model_int8 = convert_pt2e(embedding_model)
   #    # with torch.no_grad():
   #    #    segmentation_model_int8 = torch.compile(segmentation_model_int8)
   #    torch.ao.quantization.move_exported_model_to_eval(embedding_model_int8)
   #    torch.ao.quantization.allow_exported_model_train_eval(embedding_model_int8)
   #    return embedding_model_int8
   # else: # both
   #    segmentation_model_int8 = convert_pt2e(segmentation_model)
   #    embedding_model_int8 = convert_pt2e(embedding_model)
   #    torch.ao.quantization.move_exported_model_to_eval(segmentation_model_int8)
   #    torch.ao.quantization.move_exported_model_to_eval(embedding_model_int8)
   #    torch.ao.quantization.allow_exported_model_train_eval(segmentation_model_int8)
   #    torch.ao.quantization.allow_exported_model_train_eval(embedding_model_int8)
   #    # with torch.no_grad():
   #    #    segmentation_model_int8 = torch.compile(segmentation_model_int8)
   #    #    embedding_model_int8 = torch.compile(embedding_model_int8)
   #    return segmentation_model_int8, embedding_model_int8
     
# class QuantizedLightningModule(pl.LightningModule):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.quant = QuantStub()
#         self.dequant = DeQuantStub()

#     def forward(self, x):
#         x = self.quant(x)
#         x = self.model(x)
#         x = self.dequant(x)
#         return x

#     def training_step(self, batch, batch_idx):
#         return self.model.training_step(batch, batch_idx)

#     def validation_step(self, batch, batch_idx):
#         return self.model.validation_step(batch, batch_idx)

#     def test_step(self, batch, batch_idx):
#         return self.model.test_step(batch, batch_idx)

#     def configure_optimizers(self):
#         return self.model.configure_optimizers()
     
#     def train_dataloader(self):
#        return self.model.train_dataloader()
   
#     def val_dataloader(self):
#        return self.model.val_dataloader()

class QuantizedLightningModule(Model):
   def __init__(self, model):
      super().__init__()
      self.model = model
      self.task = model.task
      self.quant = QuantStub()
      self.dequant = DeQuantStub()
   
   def forward(self, x):
      x = self.quant(x)
      x = self.model(x)
      x = self.dequant(x)
      return x

class ModelType(Enum):
   Segmentation = 1
   Embedding = 2
def quantization_aware_training(model_name, max_epochs=1, batch_size=32, model_type=ModelType.Embedding):
   registry.load_database("AMI-diarization-setup/pyannote/database.yml")
   ami = registry.get_protocol("AMI.SpeakerDiarization.mini")
   
   # Only works for pyannote models
   model = Model.from_pretrained(model_name)
   finetuned_fp32 = deepcopy(model)
   finetuned_fp32 = finetuned_fp32.train()
   
   task = None
   if model_type == ModelType.Segmentation:
      task = Segmentation(ami, duration=5)
   elif model_type == ModelType.Embedding:
      task = SpeakerEmbedding(ami, duration=5)
   task.setup()
   # task._train.model = finetuned_fp32
      
   finetuned_fp32.task = task
   
   finetuned_fp32 = QuantizedLightningModule(finetuned_fp32)
   
   # Prepare model
   finetuned_fp32.eval()
   finetuned_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
   # finetuned_fp32 = fuse_all_conv_and_batchnorm_layers(finetuned_fp32)
   finetuned_fp32 = torch.ao.quantization.prepare_qat(finetuned_fp32.train())
   
   #trainer = pl.Trainer(max_epochs=max_epochs)
   # trainer.fit(finetuned_fp32)
   #train_dataloader = task.train_dataloader()
   #val_dataloader = task.val_dataloader()
   trainer = Trainer(max_epochs=max_epochs)
   #trainer.fit(finetuned_fp32, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
   trainer.fit(finetuned_fp32)
   
   finetuned_fp32.eval()
   # quantization_scale, quantization_zero_point = finetuned_fp32.qconfig.activation()
   finetuned_int8 = torch.quantization.convert(finetuned_fp32)
   
   # print(finetuned_int8)
   
   
   #print(finetuned_int8)
   
   # finetuned_int8 = ModuleWithQuantStubs(finetuned_int8, qconfig=finetuned_fp32.qconfig, quant_before_dequant=True)
   # wfinetuned_int8 = finetuned_fp32
   
   return finetuned_int8