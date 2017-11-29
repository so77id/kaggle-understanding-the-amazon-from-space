GPU=false
CUDA_VISIBLE_DEVICES=0

# DENSENET
make run-predict GPU=$GPU CONFIG_FILE=./configs/experimental-configs/densenet_121-24-11-7-config.json CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

make run-predict GPU=$GPU CONFIG_FILE=./configs/experimental-configs/densenet_121-25-11-1-config.json CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# RESNET
make run-predict GPU=$GPU CONFIG_FILE=./configs/experimental-configs/resnet_50-24-11-4-config.json CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# INCEPTION
make run-predict GPU=$GPU CONFIG_FILE=./configs/experimental-configs/inception-v4-23-11-7-config.json CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

make run-predict GPU=$GPU CONFIG_FILE=./configs/experimental-configs/inception-v4-23-11-9-config.json CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

make run-predict GPU=$GPU CONFIG_FILE=./configs/experimental-configs/inception-v4-24-11-1-config.json CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

