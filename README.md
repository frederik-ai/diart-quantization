# Real-Time Diarization

This project implements dynamic and, to some extend, static quantization (with and without QAT) of the [diart framework](https://github.com/juanmc2005/diart).

Note that the latency speed-ups which occur due to our quantization are rather negligible.

The files you will be most interested in are `quantize_models.py` and `diart_benchmark.py`.
The former file is responsible for running the quantization and performing single-inference latency benchmarks on the dynamically quantized models. 
The latter file implements a script to run benchmarks on all implmented models.
This script can be widely configured by the usage of CLI arguments.

## Preliminaries

Accept the terms and conditions of the following models

- pyannote/segmentation: https://huggingface.co/pyannote/segmentation
- pyannote/segmentation-3.0: https://huggingface.co/pyannote/segmentation-3.0
- pyannote/embedding: https://huggingface.co/pyannote/embedding

Create a huggingface access token: https://hf.co/settings/tokens. Then use huggingface-cli to log into huggingface.

On some devices you might have to run:
```bash
sudo apt-get install libportaudio2
pip install sounddevice
```

## Set-Up

### Python Environment
Run one of the two commands
```bash
conda env create --file="environment.yml"
```

```bash
pip install -r requirements.txt
```

### Dataset

Download the AMI dataset
```bash
git submodule update --init --recursive
cd AMI-diarization-setup/pyannote
sh download_ami.sh
```

Use the following script to copy the AMI files into a custom directory such that they can be used within diart (diart does not support the default file structure from pyannote)
```
sh create_diart_ami_dataset.sh
```
