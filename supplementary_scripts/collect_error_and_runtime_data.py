import numpy as np
import csv
import os
import pandas as pd
from tqdm import tqdm

from settings import pyannote_segmentation_models, pyannote_embedding_models, speechbrain_embedding_models, benchmark_out_dir

def get_average_metric_value(benchmark_path):
    """Read average metric value from the benchmark file if it has exactly 19 rows."""
    if not os.path.exists(benchmark_path):
        return np.nan
    
    with open(benchmark_path, "r") as file:
        reader = list(csv.reader(file))
        if len(reader) == 19:
            return float(reader[-1][0])
        else:
            return np.nan
         
def compute_efficiency(error_rate, latency):
   efficiency = 1 / (error_rate * latency)
   return efficiency

def get_benchmark_path(embedding_model_name, segmentation_model_name, benchmark_out_dir=f"../{benchmark_out_dir}", segmentation_model_suffix="", 
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
    benchmark_path += ".csv"
    return benchmark_path

embedding_models = []
segmentation_models = []
quantization = []
latencies = []
der_values = []
jer_values = []
der_efficiency_values = []
jer_efficiency_values = []

df_latency = pd.read_csv("diart_runtime_evaluation_results.csv", header=None)

speechbrain_embedding_models.remove("speechbrain/spkrec-ecapa-voxceleb-mel-spec")
for embedding_model in tqdm(pyannote_embedding_models + speechbrain_embedding_models):
   quantization_methods = ["none", "dynamic", "static"] if embedding_model=="pyannote/embedding" else ["none", "dynamic"]
   for segmentation_model in pyannote_segmentation_models:
      for quantization_method in quantization_methods:
         
         model_suffix = "" if quantization_method == "none" else quantization_method
         
         embedding_models.append(embedding_model)
         segmentation_models.append(segmentation_model)
         quantization.append(quantization_method)
         # DER
         der_benchmark_path = get_benchmark_path(embedding_model, segmentation_model, segmentation_model_suffix=model_suffix,
                                                 embedding_model_suffix=model_suffix, evaluation_metric="DER")
         der_value = get_average_metric_value(der_benchmark_path) if os.path.exists(der_benchmark_path) else np.nan
         der_values.append(der_value)
         # JER
         jer_benchmark_path = get_benchmark_path(embedding_model, segmentation_model, segmentation_model_suffix=model_suffix,
                                                 embedding_model_suffix=model_suffix, evaluation_metric="JER")
         jer_value = get_average_metric_value(jer_benchmark_path) if os.path.exists(jer_benchmark_path) else np.nan
         jer_values.append(jer_value)
         # Latency
         latency_row = df_latency[
            (df_latency.iloc[:, 0] == segmentation_model) &
            (df_latency.iloc[:, 1] == quantization_method) &
            (df_latency.iloc[:, 2] == embedding_model) &
            (df_latency.iloc[:, 3] == quantization_method)
         ]
         latency = latency_row.iloc[0, 4]
         latencies.append(latency)
         # DER efficiency
         der_efficiency_values.append(compute_efficiency(der_value/100, latency))
         # JER efficiency
         jer_efficiency_values.append(compute_efficiency(jer_value/100, latency))
         
df = pd.DataFrame({
    "Embedding model": embedding_models,
    "Segmentation model": segmentation_models,
    "Quantization method": quantization,
    "DER": der_values,
    "JER": jer_values,
    "Latency": latencies,
    "DER efficiency": der_efficiency_values,
    "JER efficiency": jer_efficiency_values
})
df.to_csv("diart_error_and_runtime_data.csv", index=False)