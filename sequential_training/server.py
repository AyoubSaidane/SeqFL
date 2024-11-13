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

    def flatten_model_weights(self, model):
        # Collect all the weights as flattened tensors
        flat_weights = []
        for param in model.parameters():
            flat_weights.append(param.data.view(-1).detach())
        
        # Concatenate all the flattened tensors into one 1D tensor
        flat_weights = torch.cat(flat_weights)
        
        return flat_weights.tolist()

    def cluster_clients(self, clients_weights, num_clusters, save_dendrogram, round, idx):
        # Perform hierarchical clustering
        weights = np.array(clients_weights)[:,1]
        IDs = np.array(clients_weights)[:,0]
        flattened_weights = [self.flatten_model_weights(model) for model in weights]
        distance_matrix = pdist(flattened_weights)
        linked = linkage(distance_matrix, method='ward')
        clusters = fcluster(linked, t=num_clusters, criterion='maxclust')
        nbr_clusters = len(set(clusters))

        if save_dendrogram:
            # Plot the dendrogram
            plt.figure(figsize=(10, 7))
            dendrogram(linked, labels=IDs, distance_sort='ascending')
            plt.title('Agglomerative Hierarchical Clustering Dendrogram')
            plt.xlabel('Client IDs')
            plt.ylabel('Distance')
            #plt.ylim(2700, 3000)
            
            if not os.path.exists('clusters'):
                os.makedirs('clusters')
                
            plt.savefig(f'clusters/dendrogram_round_{round}_{idx}.png')
            plt.close()
        

        sorted_indices = np.argsort(clusters)
        sorted_IDs = IDs[sorted_indices]

        clustered_IDs = [[] for _ in range(nbr_clusters)]
        for i in range(len(sorted_IDs)):
            clustered_IDs[i%nbr_clusters].append(sorted_IDs[i])

        return clustered_IDs
    


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



    def generate_combinations(self, cluster, f_byzantine):

        fraction = Fraction(f_byzantine).limit_denominator()
        # Get the numerator (p) and denominator (q)
        p = fraction.numerator
        q = fraction.denominator
        
        if p==0:
            q = len(cluster)

        # Shuffle the list
        random.shuffle(cluster)

        # Split the list into blocks of size len() numbers each
        blocks = [cluster[i:i + q] for i in range(0, len(cluster), q)]
        

        combinations_list = []
        for block in blocks:
            n = len(block)
            sequences = np.zeros((n, n), dtype=int)

            # Generate the Latin square
            for i in range(n):
                sequences[i] = np.roll(block, -i)

            # Shuffle columns
            indices = np.arange(n).tolist()
            random.shuffle(indices)
            shuffled_sequences = sequences[:, indices]
            

            # Create the dictionary
            if p == 0:
                combinations_list.append({row[0]: row[1:].tolist() for row in shuffled_sequences})
            else:
                combinations_list.append({row[0]: row[1:-p].tolist() for row in shuffled_sequences})
            
        return combinations_list


    def train_cluster(self, cluster, epochs, f_byzantine, aggregation):

        combinations_list = self.generate_combinations(cluster, f_byzantine)
        best_models = []
        for combinations in combinations_list:
            models = []

            for key in combinations.keys():  # This should be done in parallel
                model = self.clients[key].get_model()
                score = 0

                for client_ID in combinations[key]:
                    # server sends the input model
                    self.clients[client_ID].set_model(model)

                    # client training and evaluation
                    self.clients[client_ID].train(epochs)
                    score += self.clients[client_ID].evaluate()/(len(combinations[key])+1)

                    # server receives the output model
                    model = self.clients[client_ID].get_model()
                models.append([model, score])

            # Take the best model
            best_model, best_score = max(models, key=lambda x: x[1])
            best_models.append(best_model)
            
        return self.aggregation_rule(best_models, aggregation)

    def federated_train(self, rounds=1, epochs=1, num_clusters=3, f_byzantine=1/3, num_clients = 12, learning_rate = 0.01, std_dev=1, idx=''):
        performance_dict = {}
        
        if not os.path.exists('sequential_results'):
            os.makedirs('sequential_results')

        # Open the file to write performance results
        with open('sequential_results/sequential_training_performance_results_'+idx+'.txt', 'w') as file:
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
                    
                    clients_weights.append([client_id, client.get_model()])

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

                # Perform hierarchical clustering
                clusters = self.cluster_clients(clients_weights, num_clusters, self.save_dendrogram, round, idx)

                # Train clusters using PFL
                clusters_models = []
                for cluster in clusters:  # This should be done in parallel
                    # Train clients inside cluster using SFL
                    clusters_models.append(self.train_cluster(cluster, epochs, f_byzantine, self.aggregation))

                # Aggregate models
                self.global_model.load_state_dict(self.aggregation_rule(clusters_models, self.aggregation).state_dict())

            if self.save_performance:
                # Plot global and individual performance
                performance_types = ['global_performance', 'individual_performance']
                writer = SummaryWriter('sequential_results')
                for perf_type in performance_types:
                    for round_idx in range(rounds):
                        plot={}
                        for client_id in range(num_clients):
                            plot['client_'+str(client_id)] = performance_dict[client_id][perf_type][round_idx]
                        writer.add_scalars(str('sequential_byzantine_training_' + perf_type +'_'+ idx), plot, round_idx)
                writer.close()
            
                # Plot average global and individual performance
                performance_types = ['average_global_performance', 'average_individual_performance']
                scores = [global_scores, individual_scores]
                writer = SummaryWriter('sequential_results')
                for i in range(len(performance_types)):
                    for round_idx in range(rounds):  
                        plot = {performance_types[i]:scores[i][round_idx]}
                        writer.add_scalars(str('sequential_byzantine_training_' + performance_types[i]+'_'+ idx), plot, round_idx)
                writer.close()

            if self.save_model:
                # Save the global model
                if not os.path.exists('sequential_models'):
                    os.makedirs('sequential_models')
                model_path = f'sequential_models/global_model_{idx}.pth'
                torch.save(self.global_model.state_dict(), model_path)
                print(f'Global model saved at {model_path}')            

        return performance_dict
