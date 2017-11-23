import os
import shutil
import time

def get_metadata_paths(CONFIG, ARGS):
    print("Creating experiment enviroment")
    metadata_path = "{}/{}/{}".format(CONFIG.network.metadata.path, CONFIG.network.parameters.model_name, time.strftime("%d-%m-%Y"))

    n_experiment = "1"
    if os.path.exists(metadata_path):
        filenames = os.listdir(metadata_path)
        filenames = sorted(filenames)
        if len(filenames) > 0:
            last_name = int(filenames[-1])
            n_experiment = str(last_name + 1)

    metadata_path = "{}/{}".format(metadata_path, n_experiment)
    checkpoint_path =  '{}/{}'.format(metadata_path, "checkpoints")
    logs_path = "{}/{}".format(metadata_path, "logs")

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    checkpoint_path = "{}/{}".format(checkpoint_path, "model.h5")

    shutil.copy(ARGS.config_file, "{}/config.json".format(metadata_path))

    return metadata_path, checkpoint_path, logs_path