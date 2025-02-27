import os
from pyannote.audio import Model

from settings import pyannote_segmentation_models, pyannote_embedding_models, speechbrain_embedding_models

pyannote_models = pyannote_segmentation_models + pyannote_embedding_models
os.makedirs("model_architectures", exist_ok=True)

for model_name in pyannote_models:
   model = Model.from_pretrained(model_name)
   model_name_without_slash = model_name.replace("/", "-")
   
   with open(f"model_architectures/{model_name_without_slash}.txt", "w") as f:
      print(model, file=f)

for model_name in speechbrain_embedding_models:
   model = None
   if "-mel-" in model_name: # model expects mel spectrogram as input
      from speechbrain.inference.encoders import MelSpectrogramEncoder
      model = MelSpectrogramEncoder.from_hparams(source=model_name)
   else:
      from speechbrain.inference.speaker import EncoderClassifier
      model = EncoderClassifier.from_hparams(source=model_name)
   
   model_name_without_slash = model_name.replace("/", "-")
   
   with open(f"model_architectures/{model_name_without_slash}.txt", "w") as f:
      print(model, file=f)