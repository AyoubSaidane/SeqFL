import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
import torchvision.models as models
from pathlib import Path

import sys
import importlib.util



from data import labeled_idx, client_datapoints
from client import Client
import json

# Load the JSON file
with open('config.json', 'r') as json_file:
    config = json.load(json_file)

# Access values
path = config['path']
idx = config['idx']
method = config['method']
num_clients = config['num_clients']
batch_size = config['batch_size']
epochs_per_round = config['epochs_per_round']
num_rounds = config['num_rounds']
learning_rate = config['learning_rate']
num_clusters = config['num_clusters']
f_byzantine = config['f_byzantine']
num_byzantine = config['num_byzantine']
attack = config['attack']
malicious_IDs = config['malicious_IDs']
std_dev = config['std_dev']
aggregation = config['aggregation']
homogeneous_distribution = config['homogeneous_distribution']
save_distribution = config['save_distribution']
save_dendrogram = config['save_dendrogram']
save_performance = config['save_performance']
save_model = config['save_model']
device = config['device']

# Fixing seeds
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


# Load the module
spec = importlib.util.spec_from_file_location("Server", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Get the class
Server = getattr(module, 'Server')

# Load the FEMNIST dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Download and load the training and test datasets
train_dataset = EMNIST(root='data', split='balanced', train=True, download=True, transform=transform)
test_dataset = EMNIST(root='data', split='balanced', train=False, download=True, transform=transform)

# label mapping
train_label_dict = labeled_idx(train_dataset, "train")
test_label_dict = labeled_idx(test_dataset, "test")
num_classes = len(set(train_label_dict))
num_datapoints = min(min(len(lst) for lst in train_label_dict.values()),min(len(lst) for lst in test_label_dict.values()))

# Define the model
initial_model = models.resnet18(pretrained=False, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()

# Initializing the clients
clients = []

for client_id in range(num_clients):
    if homogeneous_distribution:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_idx = client_datapoints(client_id,train_label_dict, num_clients, num_datapoints, std_dev, save_distribution, idx)
        test_idx = client_datapoints(client_id,test_label_dict, num_clients, num_datapoints, std_dev, save_distribution, idx)
        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=batch_size, shuffle=False)

    client_model = models.resnet18(pretrained=False, num_classes=num_classes)
    client_model.load_state_dict(initial_model.state_dict())
    optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)

    if (client_id in malicious_IDs):
        byzantine = attack
    else:
        byzantine = 'honest'

    clients.append(Client(client_id, train_loader, test_loader, client_model, criterion, optimizer, device, byzantine))

# Initialize and run server
server_model = models.resnet18(pretrained=False, num_classes=num_classes)
server_model.load_state_dict(initial_model.state_dict())
server = Server(method, server_model, clients, device, save_dendrogram, save_performance, save_model, homogeneous_distribution, aggregation, attack)

performance_dict = server.federated_train(rounds=num_rounds,
                                          epochs=epochs_per_round,
                                          num_clusters=num_clusters,
                                          f_byzantine=f_byzantine,
                                          num_clients = num_clients,
                                          learning_rate = learning_rate,
                                          std_dev=std_dev,
                                          idx = idx)