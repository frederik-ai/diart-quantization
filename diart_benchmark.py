import argparse
from enum import Enum
import os

import torch
import torch.ao.quantization
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import Benchmark, StreamingInference, Parallelize
from diart.sources import FileAudioSource
from diart.models import SegmentationModel, EmbeddingModel, PowersetAdapter
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from pyannote.audio.pipelines.speaker_verification import SpeechBrainPretrainedSpeakerEmbedding

from settings import (
    pyannote_segmentation_models, pyannote_embedding_models, speechbrain_embedding_models, 
    quantized_models_dir, benchmark_out_dir, ami_test_wav_dir, ami_test_rttm_dir,
    audio_sample_rate, diart_runtime_evaluation_out_path
)
from quantize_models import quantize_static_pt2e_with_calibration, pt2e_quantization_aware_training
from utils import load_random_ami_data_samples

def load_dynamically_quantized_model(model_name, model_dir=quantized_models_dir):
    """
    Load a dynamically quantized model.
    
    Args:
        model_name (str): The hugginface identifier of the model. (e.g. pyannote/embedding or speechbrain/spkrec-xvect-voxceleb)
        model_dir (str): The directory where the model is saved
    """
    model_name_without_slash = model_name.replace("/", "_")
    return torch.load(f"{model_dir}/{model_name_without_slash}.pt")

def load_pt2e_diart_model(model_name, model_dir=quantized_models_dir, model_suffix=""):
    """
    Load a custom model which is exported as an executable program using torch.export.save.
    
    Args:
        model_name (str): The hugginface identifier of the model. (e.g. pyannote/embedding or speechbrain/spkrec-xvect-voxceleb)
        model_dir (str): The directory where the model is saved
    """
    model_name_without_slash = model_name.replace("/", "_")
    executable_program = torch.export.load(f"{model_dir}/{model_name_without_slash}_{model_suffix}.pt")
    return executable_program.module()

def get_benchmark_path(embedding_model_name, segmentation_model_name, benchmark_out_dir=benchmark_out_dir, segmentation_model_suffix="", 
                       embedding_model_suffix="", evaluation_metric="DER"):
    """
    Get the path to the benchmark results for a model.
    
    Args:
        embedding_model_name (str): The name of the embedding model.
        segmentation_model_name (str): The name of the segmentation model.
        benchmark_out_dir (str): The directory where the benchmark results are saved.
        
    Returns:
        str: The path to the benchmark results.
    """
    embedding_model_name_without_slash = embedding_model_name.replace("/", "_")
    segmentation_model_name_without_slash = segmentation_model_name.replace("/", "_")
    if segmentation_model_suffix:
        segmentation_model_name_without_slash += f"_{segmentation_model_suffix}"
    benchmark_directory = f"{benchmark_out_dir}/{segmentation_model_name_without_slash}"
    benchmark_path = f"{benchmark_directory}/{evaluation_metric}_{embedding_model_name_without_slash}"
    if embedding_model_suffix:
        benchmark_path += f"_{embedding_model_suffix}"
    benchmark_path += "_lr1e-6" # TODO: remove this
    benchmark_path += ".csv"
    return benchmark_path

def save_benchmark_results(embedding_model_name, segmentation_model_name, benchmark_results, metric, benchmark_out_dir=benchmark_out_dir):
    """
    Save the diart benchmark results to a CSV file.
    
    Args:
        embedding_model_name (str): The name of the embedding model.
        segmentation_model_name (str): The name of the segmentation model.
        benchmark_results (dict): The benchmark results.
        metric (str): The metric used to evaluate the benchmark results.
        benchmark_out_dir (str): The directory to save the results to.
    """
    
    segmentation_model_name_without_slash = segmentation_model_name.replace("/", "_")
    os.makedirs(f"{benchmark_out_dir}/{segmentation_model_name_without_slash}", exist_ok=True)
    embedding_model_name_without_slash = embedding_model_name.replace("/", "_")
    # ignore first row, which contains column names and convert to numeric
    # diarization_errors = benchmark_result["diarization error rate"].drop(0).apply(pd.to_numeric)
    # benchmark_result["Average DER"] = diarization_errors.mean()
    full_benchmark_path = f"{benchmark_out_dir}/{segmentation_model_name_without_slash}/{metric}_{embedding_model_name_without_slash}.csv"
    benchmark_results.to_csv(full_benchmark_path, index=False)
    
def get_pyannote_segmentation_model(segmentation_model_diart):
    # check if diart wraps the segmentation model in a PowersetAdapter
    if type(segmentation_model_diart.model) is PowersetAdapter:
        return segmentation_model_diart.model.model
    return segmentation_model_diart.model
def set_pyannote_segmentation_model(segmentation_model_diart, new_pyannote_model):
    # check if diart wraps the segmentation model in a PowersetAdapter
    if type(segmentation_model_diart.model) is PowersetAdapter:
        segmentation_model_diart.model.model = new_pyannote_model
    else:
        segmentation_model_diart.model = new_pyannote_model

def get_pyannote_embedding_model(embedding_model_diart):
    if type(embedding_model_diart.model) is SpeechBrainPretrainedSpeakerEmbedding:
        return embedding_model_diart.model.classifier_
    return embedding_model_diart.model
def set_pyannote_embedding_model(embedding_model_diart, new_pyannote_model):
    if type(embedding_model_diart.model) is SpeechBrainPretrainedSpeakerEmbedding:
        embedding_model_diart.model.classifier_ = new_pyannote_model
    else:
        embedding_model_diart.model = new_pyannote_model
    
class QuantizationMethod(Enum):
    NONE = "none"
    DYNAMIC = "dynamic"
    STATIC = "static"
    STATIC_QAT = "qat"
    
class EvaluationMetric(Enum):
    DER = "DER"
    JER = "JER"
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--segmentation_model_quantization", "-smq", type=str, default="none", required=True, help="The quantization method for the segmentation model. Options: none, dynamic, static")
    argparser.add_argument("--embedding_model_quantization", "-emq", type=str, default="none", required=True, help="The quantization method for the embedding model. Options: none, dynamic, static")
    argparser.add_argument("--static_quantization_batch_size", "-sqbs", type=int, default=16, help="The batch size for static quantization")
    argparser.add_argument("--evaluate_runtimes", "-er", action="store_true", default=False, help="Whether to evaluate the runtime of the models")
    argparser.add_argument("--segmentation_model", "-sm", type=str, default=None, help="The segmentation model to use")
    argparser.add_argument("--embedding_model", "-em", type=str, default=None, help="The embedding model to use")
    argparser.add_argument("--evaluation_metric", "-m", type=str, default=None, help="The evaluation metric to use")
    argparser.add_argument("--parralelize", "-p", action="store_true", default=False, help="Whether to parralelize the benchmark")
    args = argparser.parse_args()
    SEGMENTATION_QUANT = QuantizationMethod(args.segmentation_model_quantization)
    EMBEDDING_QUANT = QuantizationMethod(args.embedding_model_quantization)
    STATIC_QUANTIZATION_BATCH_SIZE = args.static_quantization_batch_size
    EVALUATION_METRIC = EvaluationMetric(args.evaluation_metric) if args.evaluation_metric else None
    SEGMENTATION_MODEL = args.segmentation_model
    EMBEDDING_MODEL = args.embedding_model
    
    benchmark = Benchmark(ami_test_wav_dir, ami_test_rttm_dir)
    # benchmark = Benchmark("AMI-diart/audio_files/_test", "AMI-diart/rttms/_test") # TODO: remove this
    all_embedding_models = pyannote_embedding_models + speechbrain_embedding_models
    all_embedding_models.remove("speechbrain/spkrec-ecapa-voxceleb-mel-spec") # benchmarks do not work for this model
    
    # Defined in the diart repository
    AMI_DIART_HYPERPARAMS = {
        "step": 0.5,
        "latency": 0.5,
        "tau_active": 0.507,
        "rho_update": 0.006,
        "delta_new": 1.057
    }
    
    segmentation_models = [SEGMENTATION_MODEL] if SEGMENTATION_MODEL else pyannote_segmentation_models
    embedding_models = [EMBEDDING_MODEL] if EMBEDDING_MODEL else all_embedding_models
    evaluation_metrics = [EVALUATION_METRIC] if EVALUATION_METRIC else [EvaluationMetric.DER, EvaluationMetric.JER]
    
    for segmentation_model_name in segmentation_models:
        for embedding_model_name in embedding_models:
            print(f"Segmentation model: {segmentation_model_name} [quantization: {SEGMENTATION_QUANT.value}], Embedding model: {embedding_model_name} [quantization: {EMBEDDING_QUANT.value}]")
            
            segmentation_model_diart = SegmentationModel.from_pretrained(segmentation_model_name)
            embedding_model_diart = EmbeddingModel.from_pretrained(embedding_model_name)
            # Instruct diart to initialize the Pyannote models such that they can be modified
            # Otherwise models are loaded when they are called for the first time
            segmentation_model_diart.load()
            embedding_model_diart.load()
            # Diart wraps pyannote models. These variables are references to access/modify the pyannote models in the diart wrapper.
            segmentation_model_pyannote = get_pyannote_segmentation_model(segmentation_model_diart)
            embedding_model_pyannote = get_pyannote_embedding_model(embedding_model_diart)
            
            no_quantization = SEGMENTATION_QUANT == QuantizationMethod.NONE and EMBEDDING_QUANT == QuantizationMethod.NONE
            quantization = SEGMENTATION_QUANT != QuantizationMethod.NONE or EMBEDDING_QUANT != QuantizationMethod.NONE
            
            # Dynamic quantization
            if SEGMENTATION_QUANT == QuantizationMethod.DYNAMIC:
                quantized_segmentation_model = load_dynamically_quantized_model(segmentation_model_name)
                set_pyannote_segmentation_model(segmentation_model_diart, quantized_segmentation_model)
            if EMBEDDING_QUANT == QuantizationMethod.DYNAMIC:
                quantized_embedding_model = load_dynamically_quantized_model(embedding_model_name)
                set_pyannote_embedding_model(embedding_model_diart, quantized_embedding_model)
            
            # Quantization Aware Training
            if SEGMENTATION_QUANT == QuantizationMethod.STATIC_QAT:
                waveform_segments = load_random_ami_data_samples(split="train", num_samples=48, audio_duration=5)
                segmentation_model_qat = pt2e_quantization_aware_training(segmentation_model_pyannote, embedding_model_pyannote, waveform_segments, model_to_quantize="segmentation",
                                                                          batch_size=STATIC_QUANTIZATION_BATCH_SIZE)
                set_pyannote_segmentation_model(segmentation_model_diart, segmentation_model_qat)
            if EMBEDDING_QUANT == QuantizationMethod.STATIC_QAT:
                raise ValueError("Quantization Aware Training is currently only supported for the segmentation model. Please choose a different quantization method for the embedding model.")
                
            # Static quantization
            if SEGMENTATION_QUANT == QuantizationMethod.STATIC or EMBEDDING_QUANT == QuantizationMethod.STATIC:
                waveform_segments = load_random_ami_data_samples(split="train", num_samples=48, audio_duration=5)
                
                if SEGMENTATION_QUANT == QuantizationMethod.STATIC and EMBEDDING_QUANT == QuantizationMethod.STATIC:
                    segmentation_model_ptsq, embedding_model_ptsq = quantize_static_pt2e_with_calibration(segmentation_model_pyannote, embedding_model_pyannote, waveform_segments, model_to_quantize="both",
                                                                                                batch_size=STATIC_QUANTIZATION_BATCH_SIZE)
                    set_pyannote_segmentation_model(segmentation_model_diart, segmentation_model_ptsq)
                    set_pyannote_embedding_model(embedding_model_diart, embedding_model_ptsq)
                elif SEGMENTATION_QUANT == QuantizationMethod.STATIC:
                    segmentation_model_ptsq = quantize_static_pt2e_with_calibration(segmentation_model_pyannote, embedding_model_pyannote, waveform_segments, model_to_quantize="segmentation",
                                                                                batch_size=STATIC_QUANTIZATION_BATCH_SIZE)
                    set_pyannote_segmentation_model(segmentation_model_diart, segmentation_model_ptsq)
                else:
                    embedding_model_ptsq = quantize_static_pt2e_with_calibration(segmentation_model_pyannote, embedding_model_pyannote, waveform_segments, model_to_quantize="embedding",
                                                                            batch_size=STATIC_QUANTIZATION_BATCH_SIZE)
                    set_pyannote_embedding_model(embedding_model_diart, embedding_model_ptsq)
                    
            
            device, batch_size = None, None
            if no_quantization:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                batch_size = 32
            else:
                device = torch.device("cpu")
                if SEGMENTATION_QUANT == QuantizationMethod.STATIC or EMBEDDING_QUANT == QuantizationMethod.STATIC:
                    batch_size = STATIC_QUANTIZATION_BATCH_SIZE
                else:
                    batch_size = min(os.cpu_count(), 32) # os.cpu_count() can get very high
            print(f"Device: {device}, Batch size: {batch_size}")
            
            # Prepare Paths
            embedding_model_suffix = "" if EMBEDDING_QUANT == QuantizationMethod.NONE else EMBEDDING_QUANT.value
            segmentation_model_suffix = "" if SEGMENTATION_QUANT == QuantizationMethod.NONE else SEGMENTATION_QUANT.value
            der_benchmark_path = get_benchmark_path(embedding_model_name, segmentation_model_name, evaluation_metric="DER", 
                                                    embedding_model_suffix=embedding_model_suffix, 
                                                    segmentation_model_suffix=segmentation_model_suffix)
            jer_benchmark_path = get_benchmark_path(embedding_model_name, segmentation_model_name, evaluation_metric="JER",
                                                    embedding_model_suffix=embedding_model_suffix, 
                                                    segmentation_model_suffix=segmentation_model_suffix)
            os.makedirs(os.path.dirname(der_benchmark_path), exist_ok=True)
            os.makedirs(os.path.dirname(jer_benchmark_path), exist_ok=True)
            # Run Benchmark
            config = SpeakerDiarizationConfig(
                segmentation=segmentation_model_diart,
                embedding=embedding_model_diart,
                device=device,
                **AMI_DIART_HYPERPARAMS
            )
            benchmark.batch_size = batch_size
            
            if EvaluationMetric.DER in evaluation_metrics and not os.path.exists(der_benchmark_path):
                diarization_error_rate = DiarizationErrorRate()
                if args.parralelize:
                    p_benchmark = Parallelize(benchmark, num_workers=4)
                    benchmark_result_der = p_benchmark(SpeakerDiarization, config, metric=diarization_error_rate)
                else:
                    benchmark_result_der = benchmark(SpeakerDiarization, config, metric=diarization_error_rate)
                benchmark_result_der.to_csv(der_benchmark_path, index=False)
            else:
                print("Diarization Error Rate Benchmark for this model already exists, skipping")
            if EvaluationMetric.JER in evaluation_metrics and not os.path.exists(jer_benchmark_path):
                jaccard_error_rate = JaccardErrorRate()
                if args.parralelize:
                    p_benchmark = Parallelize(benchmark, num_workers=4)
                    benchmark_result_jer = p_benchmark(SpeakerDiarization, config, metric=jaccard_error_rate)
                else:
                    enchmark_result_jer = benchmark(SpeakerDiarization, config, metric=jaccard_error_rate)
                benchmark_result_jer.to_csv(jer_benchmark_path, index=False)
            else:
                print("Jaccard Error Rate Benchmark for this model already exists, skipping")

            # Also benchmark the runtime
            if args.evaluate_runtimes:
                print("Evaluating Runtimes")
                config.device = torch.device("cpu")
                pipeline = SpeakerDiarization(config)
                pipeline_runtimes = []

                for test_file in os.listdir("AMI-diart/audio_files/calibration")[:3]:
                    audio_path = f"AMI-diart/audio_files/calibration/{test_file}"
                    audio_source = FileAudioSource(audio_path, sample_rate=audio_sample_rate)
                    streaming_inference = StreamingInference(pipeline, audio_source, batch_size=1)
                    _ = streaming_inference()
                    # Collect runtime data from the diart Chronometer
                    pipeline_runtimes.extend(streaming_inference._chrono.history)
                
                # remove outliers that come from high system load
                runtime_mean = sum(pipeline_runtimes) / len(pipeline_runtimes)
                runtime_std = torch.tensor(pipeline_runtimes).std().item()
                pipeline_runtimes = [runtime for runtime in pipeline_runtimes if runtime < (runtime_mean + 3*runtime_std)]
                
                runtime_mean = sum(pipeline_runtimes) / len(pipeline_runtimes)
                runtime_std = torch.tensor(pipeline_runtimes).std().item()
                
                with open(diart_runtime_evaluation_out_path, "a") as f:
                    data = [segmentation_model_name, SEGMENTATION_QUANT.value, 
                            embedding_model_name, EMBEDDING_QUANT.value, 
                            runtime_mean, runtime_std]
                    data_str = ",".join(map(str, data)) # convert to comma-separated string
                    f.write(data_str + "\n")
