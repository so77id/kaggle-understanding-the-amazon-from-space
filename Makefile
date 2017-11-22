# rcnn-fer Docker Makefile
PROGRAM="kaggle-amazon"

CPU_REGISTRY_URL=so77id
GPU_REGISTRY_URL=so77id
CPU_VERSION=latest-cpu
GPU_VERSION=latest-gpu
CPU_DOCKER_IMAGE=tensorflow-opencv-py3
GPU_DOCKER_IMAGE=tensorflow-opencv-py3


##############################################################################
############################# Exposed vars ###################################
##############################################################################
# enable/disable GPU usage
GPU=false
# Config file used to experiment
CONFIG_FILE="configs/config.json"
# List of cuda devises
CUDA_VISIBLE_DEVICES=0
# Name of dataset to process
PROCESS_DATASET=""

#Path to src folder
HOST_CPU_SOURCE_PATH=$(shell pwd)
HOST_GPU_SOURCE_PATH=$(shell pwd)
HOST_GPU_DATASET_PATH=/datasets/mrodriguez/kaggle

##############################################################################
############################# DOCKER VARS ####################################
##############################################################################
# COMMANDS
DOCKER_COMMAND=docker
NVIDIA_DOCKER_COMMAND=nvidia-docker

#HOST VARS
HOST_IP=127.0.0.1
HOST_TENSORBOARD_PORT=26006

#IMAGE VARS
IMAGE_TENSORBOARD_PORT=6006
IMAGE_SOURCE_PATH=/home/src
IMAGE_METADATA_PATH=$(IMAGE_SOURCE_PATH)/metadata
IMAGE_DATASET_PATH=$(IMAGE_SOURCE_PATH)/datasets


# VOLUMES

DOCKER_DISPLAY_ARGS = -e DISPLAY=${HOST_IP}:0 \
                      --volume="${HOME}/.Xauthority:/root/.Xauthority:rw" \


CPU_DOCKER_VOLUMES = --volume=$(HOST_CPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH)

GPU_DOCKER_VOLUMES = --volume=$(HOST_GPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
					 --volume=$(HOST_GPU_DATASET_PATH):$(IMAGE_DATASET_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH)


DOCKER_PORTS = -p $(HOST_IP):$(HOST_TENSORBOARD_PORT):$(IMAGE_TENSORBOARD_PORT)

# IF GPU == false --> GPU is disabled
# IF GPU == true --> GPU is enabled
ifeq ($(GPU), true)
	DOCKER_RUN_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm  $(GPU_DOCKER_VOLUMES) $(DOCKER_DISPLAY_ARGS) $(GPU_REGISTRY_URL)/$(GPU_DOCKER_IMAGE):$(GPU_VERSION)
	DOCKER_RUN_PORT_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm  $(DOCKER_PORTS) $(DOCKER_DISPLAY_ARGS) $(GPU_DOCKER_VOLUMES) $(GPU_REGISTRY_URL)/$(GPU_DOCKER_IMAGE):$(GPU_VERSION)
else
	DOCKER_RUN_COMMAND=$(DOCKER_COMMAND) run -it --rm  $(CPU_DOCKER_VOLUMES) $(DOCKER_DISPLAY_ARGS) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
	DOCKER_RUN_PORT_COMMAND=$(DOCKER_COMMAND) run -it --rm  $(DOCKER_PORTS) $(DOCKER_DISPLAY_ARGS) $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
endif



##############################################################################
############################## CODE VARS #####################################
##############################################################################
#COMMANDS
PYTHON_COMMAND=python3
EXPORT_COMMAND=export
BASH_COMMAND=bash
TENSORBOARD_COMMAND=tensorboard
WGET_COMMAND=wget
MV_COMMAND=mv


MODELS_FOLDER=./models
PREPROCESSING_FOLDER=./preprocessing

TRAIN_INCEPTION_FILE=$(MODELS_FOLDER)/inception-v4.py


CREATE_H5_FILE=$(PREPROCESSING_FOLDER)/create_h5_files.py




##############################################################################
############################ CODE COMMANDS ###################################
##############################################################################
train-inception t: excuda-devise
	@echo "[Train] Trainning..."
	@$(PYTHON_COMMAND) $(TRAIN_INCEPTION_FILE) -c $(CONFIG_FILE)


dataset d: excuda-devise
	@echo "[preprocessing] preprocessing dataset..."
	@$(PYTHON_COMMAND) $(CREATE_H5_FILE) -c $(CONFIG_FILE)

excuda-devise ecd:
ifeq ($(GPU), true)
	@echo "\t Using CUDA_VISIBLE_DEVICES: "$(CUDA_VISIBLE_DEVICES)
	@$(EXPORT_COMMAND) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
endif



##############################################################################
########################### DOCKER COMMANDS ##################################
##############################################################################
run-inception rt: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make train-inception CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) CONFIG_FILE=$(CONFIG_FILE)";

run-dataset rt: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make dataset CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) CONFIG_FILE=$(CONFIG_FILE)";

#PRIVATE
docker-print psd:
ifeq ($(GPU), true)
	@echo "[GPU Docker] Running gpu docker image..."
else
	@echo "[CPU Docker] Running cpu docker image..."
endif
