import torch
import torch.nn as nn
import numpy as np
import random
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from fractions import Fraction
import os
import json
import copy
from torch.utils.tensorboard import SummaryWriter


class Server:
    def __init__(self, method, global_model, clients, device, save_dendrogram, save_performance, save_model, homogeneous_distribution, aggregation, attack):
        self.method = method
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device= device
        self.save_dendrogram = save_dendrogram
        self.save_performance = save_performance
        self.save_model = save_model
        self.homogeneous_distribution = homogeneous_distribution
        self.aggregation = aggregation
        self.attack = attack


    def aggregation_rule(self, models_list, aggregation):
        # Initialize the aggregated model
        aggregated_model = copy.deepcopy(models_list[0])
        
        # Initialize all parameters of the new model to zeros
        for param in aggregated_model.parameters():
            nn.init.zeros_(param)
        
        # Iterate over the parameters and aggregate
        for param_idx, param in enumerate(aggregated_model.parameters()):
            for model in models_list:
                if aggregation == 'avg':
                    param.data += list(model.parameters())[param_idx].data / len(models_list)
                
        return aggregated_model


    def federated_train(self, rounds=1, epochs=1, num_clusters=3, f_byzantine=1/3, num_clients = 12, learning_rate = 0.01, std_dev=1, idx=''):
        performance_dict = {}
        
        if not os.path.exists('fully_sequential_results'):
            os.makedirs('fully_sequential_results')

        # Open the file to write performance results
        with open('fully_sequential_results/fully_sequential_training_performance_results_'+idx+'.txt', 'w') as file:
            # Write configuration details
            config_details = (f"Configuration:\n"
                              f"Algorithm: {self.method}\n"
                              f"Aggregation Method: {self.aggregation}\n"
                              f"Attack: {self.attack}\n"
                              f"Fraction of Byzantine Clients: {f_byzantine}\n"
                              f"Number of clients: {num_clients}\n"
                              f"Number of Clusters: {num_clusters}\n"
                              f"Rounds: {rounds}\n"
                              f"Standard deviation: {std_dev}\n"
                              f"Homogeneous distribution: {self.homogeneous_distribution}\n"
                              f"Epochs per round: {epochs}\n"
                              f"Learning rate: {learning_rate}\n"
                              f"Device: {self.device}\n\n")
            file.write(config_details)
            print(config_details.strip())

            round_scores = []
            client_order = list(range(num_clients))
            for round in range(rounds):
                # Initialize metrics for the current round

                random.shuffle(client_order)
                print(client_order)
                for client_ID in client_order:
                    self.clients[client_ID].set_model(self.global_model)
                    self.clients[client_ID].train(epochs)
                    self.global_model = self.clients[client_ID].get_model()
                
                round_score = self.clients[client_order[-1]].evaluate()
                round_scores.append(round_score)

                round_result = f'Round {round+1}, Accuracy: {round_score}\n'
                print(round_result.strip())
                file.write(round_result)

            if self.save_performance:
            
                writer = SummaryWriter('fully_sequential_results')
                for round_idx in range(rounds):
                    writer.add_scalars(str('fully_sequential_byzantine_training_' +'_'+ idx), {'' : round_scores[round_idx]}, round_idx)
                writer.close()

            if self.save_model:
                # Save the global model
                if not os.path.exists('fully_sequential_models'):
                    os.makedirs('fully_sequential_models')
                model_path = f'fully_sequential_models/global_model_{idx}.pth'
                torch.save(self.global_model.state_dict(), model_path)
                print(f'Global model saved at {model_path}')            

        return performance_dict
