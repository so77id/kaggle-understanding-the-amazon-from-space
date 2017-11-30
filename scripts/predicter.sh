GPU=false
CUDA_VISIBLE_DEVICES=0

# DENSENET
make run-predict GPU=$GPU CONFIG_FILE=./configs/experimental-configs/densenet_121-30-11-7-config.json CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# RESNET
make run-predict GPU=$GPU CONFIG_FILE=./configs/experimental-configs/resnet_50-30-11-5-config.json CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# INCEPTION
make run-predict GPU=$GPU CONFIG_FILE=./configs/experimental-configs/inception-v4-30-11-3-config.json CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

