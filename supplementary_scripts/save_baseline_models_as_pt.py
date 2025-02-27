import os
import torch
from pyannote.audio import Model
from speechbrain.inference.encoders import MelSpectrogramEncoder
from speechbrain.inference.speaker import EncoderClassifier

from settings import pyannote_segmentation_models, pyannote_embedding_models, speechbrain_embedding_models

TARGET_DIR = "baseline_models"
os.makedirs(TARGET_DIR, exist_ok=True)

all_models = pyannote_segmentation_models + pyannote_embedding_models + speechbrain_embedding_models

# Save Pyannote models
for model_name in all_models:
   model = None
   if "pyannote" in model_name:
      model = Model.from_pretrained(model_name)
   elif "speechbrain" in model_name:
      if "-mel-" in model_name:
         model = MelSpectrogramEncoder.from_hparams(source=model_name)
      else:
         model = EncoderClassifier.from_hparams(source=model_name)
         inference_function = model.encode_batch
   else:
      raise ValueError(f"Unsupported model: {model_name}")
   
   # Save quantized model to .pt and onnx
   model_name_without_slash = model_name.replace("/", "_")
   torch.save(model, f"{TARGET_DIR}/{model_name_without_slash}.pt")