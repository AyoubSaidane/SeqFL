import random
import numpy as np
import torch
import json
import os


def convert_np_ints(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.device):
        return str(obj)
    raise TypeError

method_list = ['sequential']
epochs_per_round_list = [1]
std_dev_list = [1]
f_byzantine = 1/3
num_clients = 12
num_byzantine = int(f_byzantine * num_clients)
num_rounds = 3


for method in method_list:
    for epochs_per_round in epochs_per_round_list:
        for std_dev in std_dev_list:
            config = {
                "idx": method + '_epochs_' + str(epochs_per_round) + '_std_' + str(std_dev)+ '_rounds_' + str(num_rounds),
                "method": method,
                "path": '/home/cc/benchmark/'+method+'_training/server.py',
                "num_clients": num_clients,
                "batch_size": 32,
                "epochs_per_round": epochs_per_round,
                "num_rounds": num_rounds,
                "learning_rate": 0.1,
                "num_clusters": 3,
                "f_byzantine": f_byzantine,
                "num_byzantine": num_byzantine,
                "attack": 'honest',
                "malicious_IDs": random.sample(list(np.arange(num_clients)), num_byzantine),
                "std_dev": std_dev,
                "aggregation": 'avg',
                "homogeneous_distribution": False,
                "save_distribution": True,
                "save_dendrogram": True,
                "save_performance": True,
                "save_model": True,
                "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            }

            with open('config.json', 'w') as json_file:
                json.dump(config, json_file, indent=4, default=convert_np_ints)

            os.system('python main.py')