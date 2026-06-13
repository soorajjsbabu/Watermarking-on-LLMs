# # analyze_results.py
# import pandas as pd
# import numpy as np
# from scipy.spatial.distance import euclidean
# import json

# def load_embeddings(csv_file):
#     print("load")
#     df = pd.read_csv(csv_file)
#     # Convert string embeddings back to numpy arrays
#     embeddings = df['embedding'].apply(lambda x: np.array(json.loads(x)))
#     return embeddings.values

# def analyze_system(embeddings, system_name):
#     print("analyze")
#     centroid = np.mean(embeddings, axis=0)
#     distances = [euclidean(emb, centroid) for emb in embeddings]
    
#     return {
#         'system': system_name,
#         'centroid': centroid,
#         'max_radius': max(distances),
#         'min_distance': min(distances),
#         'mean_distance': np.mean(distances),
#         'std_distance': np.std(distances)
#     }

# # Load and analyze each system
# non_watermarked = load_embeddings('non_watermarked_results.csv')
# hard_watermarked = load_embeddings('hard_watermarked_results.csv')
# soft_watermarked = load_embeddings('soft_watermarked_results.csv')

# # Analyze each system
# results = []
# for embeddings, name in [(non_watermarked, 'Non-watermarked'), 
#                         (hard_watermarked, 'Hard-watermarked'),
#                         (soft_watermarked, 'Soft-watermarked')]:
#     results.append(analyze_system(embeddings, name))

# print(results)

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from collections import Counter
import json

def load_and_process_embeddings(csv_file):
    df = pd.read_csv(csv_file)
    embeddings = df['embedding'].apply(lambda x: np.array(json.loads(x)))
    return embeddings.values

def analyze_distances(embeddings, centroid):
    distances = [euclidean(emb, centroid) for emb in embeddings]
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Categorize distances relative to mean
    threshold = std_dist / 2
    categories = {
        'larger': sum(1 for d in distances if d > mean_dist + threshold),
        'smaller': sum(1 for d in distances if d < mean_dist - threshold),
        'approximate': sum(1 for d in distances if abs(d - mean_dist) <= threshold)
    }
    
    return {
        'distances': distances,
        'mean': mean_dist,
        'std': std_dist,
        'max': max(distances),
        'min': min(distances),
        'categories': categories
    }

def analyze_system(embeddings, system_name):
    centroid = np.mean(embeddings, axis=0)
    stats = analyze_distances(embeddings, centroid)
    
    return {
        'system': system_name,
        'centroid': centroid,
        'statistics': stats
    }

def plot_distance_distribution(results):
    plt.figure(figsize=(12, 6))
    for system in results:
        plt.hist(system['statistics']['distances'], 
                alpha=0.5, 
                label=system['system'],
                bins=20)
    plt.xlabel('Distance from Centroid')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances from Centroid')
    plt.legend()
    plt.show()

# Main analysis
systems = [
    ('non_watermarked_results.csv', 'Non-watermarked'),
    ('hard_watermarked_results.csv', 'Hard-watermarked'),
    ('soft_watermarked_results.csv', 'Soft-watermarked')
]

results = []
for file_path, name in systems:
    embeddings = load_and_process_embeddings(file_path)
    result = analyze_system(embeddings, name)
    results.append(result)
    
    print(f"\n{name} System Analysis:")
    print(f"Mean Distance: {result['statistics']['mean']:.4f}")
    print(f"Std Deviation: {result['statistics']['std']:.4f}")
    print(f"Max Distance: {result['statistics']['max']:.4f}")
    print(f"Min Distance: {result['statistics']['min']:.4f}")
    print("\nDistance Categories:")
    for cat, count in result['statistics']['categories'].items():
        print(f"{cat.capitalize()}: {count}")

# Plot distributions
plot_distance_distribution(results)

# Calculate cross-system distances
for i, sys1 in enumerate(results):
    for sys2 in results[i+1:]:
        dist = euclidean(sys1['centroid'], sys2['centroid'])
        print(f"\nDistance between {sys1['system']} and {sys2['system']} centroids: {dist:.4f}")
