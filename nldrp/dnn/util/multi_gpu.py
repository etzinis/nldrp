import json
import os

import torch

from nldrp.dnn.config import DNN_BASE_PATH

GPU_FILE = os.path.join(DNN_BASE_PATH, "GPU.json")
GPU_ID = 0

GPU_COUNT = torch.cuda.device_count()
if not os.path.exists(GPU_FILE):
    with open(GPU_FILE, 'w') as f:

        config = {
            "count": GPU_COUNT,
            "current": 1
        }
        json.dump(config, f)

else:
    with open(GPU_FILE, 'r') as f:
        config = json.load(f)
        config["count"] = GPU_COUNT
    with open(GPU_FILE, 'w') as f:
        json.dump(config, f)


def write_config(config):
    # Writing JSON data
    with open(GPU_FILE, 'w') as f:
        json.dump(config, f)


def read_config():
    with open(GPU_FILE, 'r') as f:
        config = json.load(f)
        return config


def get_gpu_id():
    return GPU_ID


def get_new_gpu_id():
    global GPU_ID

    config = read_config()
    if config["count"] == 1:
        GPU_ID = 0
    else:
        try:
            config["current"] = (config["current"] + 1) % config["count"]
            write_config(config)
            GPU_ID = config["current"]
        except:
            GPU_ID = 0

    return GPU_ID
