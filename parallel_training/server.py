import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json
import copy
import pdb
from scipy.spatial.distance import pdist
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
        self.homogeneous_distribution=homogeneous_distribution
        self.aggregation = aggregation
        self.attack= attack


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

    def federated_train(self, rounds=1, epochs=1, num_clusters=3, f_byzantine=1/3, num_clients = 12,learning_rate = 0.01, std_dev=1, idx=''):
        performance_dict = {}

        if not os.path.exists('parallel_results'):
            os.makedirs('parallel_results')

        # Open the file to write performance results
        with open('parallel_results/parallel_training_performance_results_'+idx+'.txt', 'w') as file :
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

            global_scores = []
            individual_scores = []
            
            for round in range(rounds):
                # Initialize metrics for the current round
                global_score = 0
                individual_score = 0
                clients_weights = []

                for client in self.clients:  # This should be done in parallel
                    client_id = client.get_ID()
                    client.set_model(self.global_model)

                    client_global_score = client.evaluate()
                    global_score += client_global_score / len(self.clients)
                
                    client.train(epochs)
                    
                    client_individual_score = client.evaluate()
                    individual_score += client_individual_score / len(self.clients)
                    
                    clients_weights.append(client.get_model())

                    # Update the performance dictionary
                    if client_id not in performance_dict:
                        performance_dict[client_id] = {
                            'global_performance': [],
                            'individual_performance': []
                        }
                    performance_dict[client_id]['global_performance'].append(client_global_score)
                    performance_dict[client_id]['individual_performance'].append(client_individual_score)

                # Print and save round results
                global_scores.append(global_score)
                individual_scores.append(individual_score)
                round_result = f'Round {round+1}, Average Global Accuracy: {global_score}, Average Individual Accuracy: {individual_score}\n'
                print(round_result.strip())
                file.write(round_result)
                
                #pdb.set_trace()

                # Aggregate models
                self.global_model.load_state_dict(self.aggregation_rule(clients_weights, self.aggregation).state_dict())


            if self.save_performance:
                # Plot global and individual performance
                performance_types = ['global_performance', 'individual_performance']
                writer = SummaryWriter('parallel_results')
                for perf_type in performance_types:
                    for round_idx in range(rounds):
                        plot={}
                        for client_id in range(num_clients):
                            plot['client_'+str(client_id)] = performance_dict[client_id][perf_type][round_idx]
                        writer.add_scalars(str('parallel_byzantine_training_' + perf_type +'_'+ idx), plot, round_idx)
                writer.close()
            
                # Plot average global and individual performance
                performance_types = ['average_global_performance', 'average_individual_performance']
                scores = [global_scores, individual_scores]
                writer = SummaryWriter('parallel_results')
                for i in range(len(performance_types)):
                    for round_idx in range(rounds):  
                        plot = {performance_types[i]:scores[i][round_idx]}
                        writer.add_scalars(str('parallel_byzantine_training_' + performance_types[i]+'_'+ idx), plot, round_idx)
                writer.close()
            
            if self.save_model:
                # Save the global model
                if not os.path.exists('parallel_models'):
                    os.makedirs('parallel_models')
                model_path = f'parallel_models/global_model_{idx}.pth'
                torch.save(self.global_model.state_dict(), model_path)
                print(f'Global model saved at {model_path}')  

        return performance_dict
