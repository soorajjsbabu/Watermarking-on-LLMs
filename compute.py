import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import json

def load_embeddings(csv_file):
    df = pd.read_csv(csv_file)
    embeddings = df['embedding'].apply(lambda x: np.array(json.loads(x)))
    return embeddings.values

def compute_distances_from_centroid(embeddings, centroid):
    return [euclidean(emb, centroid) for emb in embeddings]

# Load embeddings
non_watermarked = load_embeddings('non_watermarked_results.csv')
hard_watermarked = load_embeddings('hard_watermarked_results.csv')
soft_watermarked = load_embeddings('soft_watermarked_results.csv')

# Compute non-watermarked centroid
nw_centroid = np.mean(non_watermarked, axis=0)

# Compute distances from non-watermarked centroid
nw_distances = compute_distances_from_centroid(non_watermarked, nw_centroid)
hw_distances = compute_distances_from_centroid(hard_watermarked, nw_centroid)
sw_distances = compute_distances_from_centroid(soft_watermarked, nw_centroid)

# Plot histogram
plt.figure(figsize=(12, 6))
plt.hist(nw_distances, bins=20, alpha=0.5, label='Non-watermarked', color='blue')
plt.hist(hw_distances, bins=20, alpha=0.5, label='Hard-watermarked', color='orange')
plt.hist(sw_distances, bins=20, alpha=0.5, label='Soft-watermarked', color='green')

plt.xlabel('Distance from Non-watermarked Centroid')
plt.ylabel('Frequency')
plt.title('Distribution of Distances from Non-watermarked Centroid')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
