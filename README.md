# kaggle-Planet-Understanding-the-Amazon-from-Space
https://www.kaggle.com/c/planet-understanding-the-amazon-from-space

Classes:

0  : agriculture
1  : artisinal_mine
2  : bare_ground
3  : blooming
4  : blow_down
5  : clear
6  : cloudy
7  : conventional_mine
8  : cultivation
9  : habitation
10 : haze
11 : partly_cloudy
12 : primary
13 : road
14 : selective_logging
15 : slash_burn
16 : water


##Requirements
* Docker
* Make

## Parameters
* GPU=`true|false`
* CUDA_VISIBLE_DEVICES=`GPU_ID`
* CONFIG_FILE=`path_to_config_file`

# Commands

### run-train
Run train experiment for specific network
`make run-train CONFIG_FILE=./configs/densenet-121-config.json GPU=true CUDA_VISIBLE_DEVICES=0` --> Densenet configuration
`make run-train CONFIG_FILE=./configs/inception-v4-config.json GPU=true CUDA_VISIBLE_DEVICES=0` --> Inception configuration
`make run-train CONFIG_FILE=./configs/resnet-50-config.json GPU=true CUDA_VISIBLE_DEVICES=0` --> Resnet configuration

### run-predict
Run predict for specefic trained network
`make run-predict CONFIG_FILE=./configs/densenet-121-config.json GPU=true CUDA_VISIBLE_DEVICES=0` --> Densenet configuration
`make run-predict CONFIG_FILE=./configs/inception-v4-config.json GPU=true CUDA_VISIBLE_DEVICES=0` --> Inception configuration
`make run-predict CONFIG_FILE=./configs/resnet-50-config.json GPU=true CUDA_VISIBLE_DEVICES=0` --> Resnet configuration

### run-dataset
Create h5 files with datasets
`make run-dataset CONFIG_FILE=./configs/densenet-121-config.json` --> Densenet configuration
`make run-dataset CONFIG_FILE=./configs/inception-v4-config.json` --> Inception configuration
`make run-dataset CONFIG_FILE=./configs/resnet-50-config.json` --> Resnet configuration


### run-ensemble
Create prediction with ensemble using the predictions with networks specified in config file
`make run-ensemble CONFIG_FILE=./configs/ensemble-config.json GPU=true CUDA_VISIBLE_DEVICES=0`


# Authors
@marcostx
@barbarabenato
@Brenolleite
@so77id
