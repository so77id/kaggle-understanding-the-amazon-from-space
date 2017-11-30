GPU=true
CUDA_VISIBLE_DEVICES=2

# DENSENET
make run-predict GPU=true CONFIG_FILE=./configs/predict_train/densenet-7.json CUDA_VISIBLE_DEVICES=2

# RESNET
make run-predict GPU=true CONFIG_FILE=./configs/predict_train/densenet-1.json CUDA_VISIBLE_DEVICES=2

# INCEPTION
make run-predict GPU=true CONFIG_FILE=./configs/predict_train/inception-7.json CUDA_VISIBLE_DEVICES=2

