import torch
import torchaudio
from pyannote.database import registry, FileFinder

from settings import audio_sample_rate, use_mono_audio

def load_pyannote_ami_protocol():
   preprocessors = {'audio': FileFinder()}
   registry.load_database('AMI-diarization-setup/pyannote/database.yml')
   protocol = registry.get_protocol('AMI.SpeakerDiarization.only_words', preprocessors=preprocessors)
   return protocol

def load_random_audio_segment(file, audio_duration, target_sample_rate=audio_sample_rate, convert_to_mono=use_mono_audio):
   waveform, sample_rate = torchaudio.load(file)
   if sample_rate != target_sample_rate:
      resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
      waveform = resample(waveform)
   waveform = waveform.mean(dim=0, keepdim=True) if convert_to_mono else waveform
   # sample a random segment of the audio
   total_num_samples = waveform.size(1)
   target_num_samples = int(audio_duration * target_sample_rate)
   waveform_segment_start = torch.randint(0, total_num_samples - target_num_samples, (1,))
   waveform_segment_end = waveform_segment_start + target_num_samples
   waveform_segment = waveform[:, waveform_segment_start:waveform_segment_end]
   return waveform_segment

def load_random_ami_data_samples(split="test", num_samples=16, audio_duration=5):
   protocol = load_pyannote_ami_protocol()
   
   def get_data_iterator():
      if split == "test":
         return protocol.test()
      elif split == "train":
         return protocol.train()
      return None
   data = get_data_iterator()

   num_audio_channels = 1 if use_mono_audio else 2
   samples = torch.zeros(num_samples, num_audio_channels, audio_duration * audio_sample_rate)
   
   while num_samples > 0:
      dataset_sample = next(data, None)
      # If iterator is consumed (meaning we traversed the whole split), create a new iterator
      if dataset_sample is None:
         data = get_data_iterator()
         dataset_sample = next(data)
      audio_file = dataset_sample['audio']
      audio_segment = load_random_audio_segment(audio_file, audio_duration)
      samples[num_samples - 1] = audio_segment
      num_samples -= 1
   return samples

   # Old code that could only return as many sampels as there are audio files in the split
   # for i in range(num_samples):
   #    audio_file = next(data)['audio']
   #    audio_segment = load_random_audio_segment(audio_file, audio_duration)
   #    samples[i] = audio_segment
   # return samples
