#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

save_dir="/Users/jongbeomkim/Desktop/workspace/Classifier-Free-Guidance/samples"

python3 ../sample.py\
    --model_params="/Users/jongbeomkim/Documents/classifier-guidance/unet-channels=128/epoch=28-val_loss=0.0303.pth"\
    --save_path="$save_dir/classifier_scale=$classifier_scale-1.jpg"\
    --batch_size=2\
