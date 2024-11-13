from collections import defaultdict
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import pickle
import os


def labeled_idx(dataset, mode):
    # Check if the dictionary has already been saved
    filename = f'{mode}_label_dict.pkl'
    try:
        with open(filename, 'rb') as f:
            label_dict = pickle.load(f)
        print(f"Loaded {mode} label dictionary from {filename}")
        return label_dict
    except FileNotFoundError:
        print(f"{filename} not found. Calculating label dictionary...")

    # Calculate and save the label dictionary
    label_dict = defaultdict(list)
    idx = 0
    for (data, target) in dataset:
        label_dict[target].append(idx)
        idx += 1

    with open(filename, 'wb') as f:
        pickle.dump(label_dict, f)
        print(f"Saved {mode} label dictionary to {filename}")

    return label_dict


def client_data_distribution(client_id,num_classes,num_clients,std_dev,save_distribution, idx):
    mean = (client_id+1/2)*num_classes/num_clients

    # Initialize the list to store the probabilities
    probabilities = []

    # Calculate the probability for the interval (-inf, 1)
    prob_neg_inf_to_1 = stats.norm.cdf(1, mean, std_dev)
    probabilities.append(prob_neg_inf_to_1)

    # Calculate the probabilities for the intervals (1, 2), (2, 3), ..., (num_classes-1, num_classes)
    for i in range(1, num_classes-1):
        prob_i_to_i_plus_1 = stats.norm.cdf(i + 1, mean, std_dev) - stats.norm.cdf(i, mean, std_dev)
        probabilities.append(prob_i_to_i_plus_1)

    # Calculate the probability for the interval (num_classes, +inf)
    prob_num_classes_to_inf = 1 - stats.norm.cdf(num_classes-1, mean, std_dev)
    probabilities.append(prob_num_classes_to_inf)

    if save_distribution:
        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(1, num_classes + 1), probabilities)
        plt.xlabel('Class Intervals')
        plt.ylabel('Probability')
        plt.title(f'Gaussian Distribution Probabilities (mean={mean}, std_dev={std_dev})')
        plt.xticks(ticks=np.arange(1, num_classes + 1), labels=[f'(-inf, 1)'] + [f'({i}, {i+1})' for i in range(1, num_classes-1)] + [f'({num_classes}, +inf)'])
        plt.grid(True)

        # Create results directory if it doesn't exist
        if not os.path.exists('graphs'):
            os.makedirs('graphs')
            
        plt.savefig(f'graphs/distribution_{client_id}_{idx}.png')
        plt.close()
        
    return probabilities

def client_datapoints(client_id,label_dict, num_clients, num_datapoints, std_dev, save_distribution, idx):
    proportions = client_data_distribution(client_id, len(set(label_dict)), num_clients, std_dev, save_distribution, idx)
    selected_indices = []
    for label in set(label_dict):
        indices = label_dict[label]
        num_samples = int(proportions[label] * num_datapoints)
        selected_indices.extend(np.random.choice(indices, num_samples, replace=False))
    return selected_indices
