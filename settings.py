import os

# Pyannote raises a specific warning that is not useful
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The 'delim_whitespace' keyword in pd.read_csv")

# huggingface token
from dotenv import load_dotenv
load_dotenv()
HUGGINGFACE_AUTH_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN")

# Models
pyannote_segmentation_models = ["pyannote/segmentation", "pyannote/segmentation-3.0"]
pyannote_embedding_models = ["pyannote/embedding", "pyannote/wespeaker-voxceleb-resnet34-LM"]	
speechbrain_embedding_models = ["speechbrain/spkrec-xvect-voxceleb", "speechbrain/spkrec-ecapa-voxceleb", "speechbrain/spkrec-ecapa-voxceleb-mel-spec",
                                "speechbrain/spkrec-resnet-voxceleb"]
nivida_nemo_embedding_models = ["nvidia/speakerverification_en_titanet_large"]

# Paths
benchmark_out_dir = "diart_benchmarks"
quantized_models_dir = "quantized_models"
ami_test_wav_dir = "AMI-diart/audio_files/test" # directory where wav files of ami test set are stored
ami_test_rttm_dir = "AMI-diart/rttms/test" # directory where rttm files of ami test set are stored
ami_calibration_wav_dir = "AMI-diart/audio_files/calibration" # directory where wav files of ami calibration set are stored
runtime_evaluation_out_path = "runtime_evaluation_results.csv"
diart_runtime_evaluation_out_path = "diart_runtime_evaluation_results.csv"

# Miscellanous
audio_sample_rate = 16000 # according to the pyannote paper (https://arxiv.org/abs/2104.04045)
use_mono_audio = True