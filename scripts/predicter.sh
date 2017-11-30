GPU=true
CUDA_VISIBLE_DEVICES=2

# DENSENET
make run-predict-train GPU=true CONFIG_FILE=./configs/predict_train/densenet-7.json CUDA_VISIBLE_DEVICES=0

# RESNET
make run-predict-train GPU=true CONFIG_FILE=./configs/predict_train/densenet-1.json CUDA_VISIBLE_DEVICES=0

# INCEPTION
make run-predict-train GPU=true CONFIG_FILE=./configs/predict_train/inception-7.json CUDA_VISIBLE_DEVICES=0

