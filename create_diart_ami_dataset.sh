mkdir -p AMI-diart/rttms/test AMI-diart/audio_files/test
mkdir -p AMI-diart/rttms/train AMI-diart/audio_files/train
# copy each file from folder AMI-diarization-setup/only_words/rttms/test to AMI-diart/rttms/test
cp AMI-diarization-setup/only_words/rttms/test/* AMI-diart/rttms/test
# copy each file from folder AMI-diarization-setup/only_words/rttms/train to AMI-diart/rttms/train
cp AMI-diarization-setup/only_words/rttms/train/* AMI-diart/rttms/train

# for each file in AMI-diart/rttms/test, copy the wav file with the same basename from AMI-diarization-setup/audio/test to AMI-diart/audio_files/test
for rttm in AMI-diart/rttms/test/*; do
    base=$(basename $rttm .rttm)
    cp AMI-diarization-setup/pyannote/amicorpus/$base/audio/$base.Mix-Headset.wav AMI-diart/audio_files/test/$base.wav
done

# for the first 10 files in AMI-diart/rttms/train, copy the wav file with the same basename from AMI-diarization-setup/audio/train to AMI-diart/audio_files/train
# these files can be used for calibration purposes
for rttm in $(ls AMI-diart/rttms/train | head -n 10); do
    base=$(basename $rttm .rttm)
    cp AMI-diarization-setup/pyannote/amicorpus/$base/audio/$base.Mix-Headset.wav AMI-diart/audio_files/train/$base.wav
done

# Add calibration data for quantization
mkdir -p AMI-diart/rttms/calibration AMI-diart/audio_files/calibration
python supplementary_scripts/create_calibration_data.py