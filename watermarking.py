import pandas as pd
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from hashlib import sha256
from scipy.spatial.distance import cosine
import torch
import csv
import json
from scipy.spatial.distance import euclidean

# Configuration
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.1:8b"
SEED = 42
GAMMA = 0.5  # Green list size ratio
DELTA = 2.0  # Soft watermarking delta
BATCH_SIZE = 32  # Number of entries per batch
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # For semantic similarity

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize SentenceTransformer on the GPU
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load MS MARCO Dataset
def load_collection(file_path):
    """
    Load MS MARCO collection.tsv file into a pandas DataFrame.
    """
    data = pd.read_csv(file_path, sep="\t", names=["id", "text"], quoting=3)
    return data

# Generate Token Lists
def generate_token_lists(vocab_size, seed):
    """
    Partition the vocabulary into green and red lists.
    """
    np.random.seed(seed)
    indices = np.arange(vocab_size)
    np.random.shuffle(indices)
    split = int(GAMMA * vocab_size)
    return indices[:split], indices[split:]  # Green list, Red list

# Watermarking Functions
def apply_watermark(logits, vocab_size, seed, watermark_type="hard", delta=2.0):
    """
    Modify logits for watermarking.
    """
    green_list, red_list = generate_token_lists(vocab_size, seed)
    if watermark_type == "hard":
        logits[red_list] = -np.inf  # Block red list tokens
    elif watermark_type == "soft":
        logits[green_list] += delta  # Boost green list tokens
    return logits

def generate_text_with_watermark(prompt, watermark_type=None, delta=2.0):
    """
    Generate text with watermarking via Ollama API and process streaming responses.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_length": 50,
    }
    response = requests.post(OLLAMA_API_URL, json=payload, stream=True)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} {response.text}")

    # Accumulate responses
    generated_text = ""
    try:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                line_data = json.loads(line)
                generated_text += line_data.get("response", "")
                if line_data.get("done", False):
                    break
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON line: {e}\nLine content: {line}")

    vocab_size = 50000 
    logits = np.random.rand(vocab_size)  

    if watermark_type in ["hard", "soft"]:
        logits = apply_watermark(logits, vocab_size, SEED, watermark_type, delta)
        sampled_token_id = np.argmax(logits)  # Greedy sampling
        generated_text += f" [Token ID: {sampled_token_id}]"

    return generated_text

# Semantic Similarity
def calculate_semantic_similarity(text1, text2):
    """
    Compute semantic similarity between two texts using embeddings.
    """
    embedding1 = EMBEDDING_MODEL.encode(text1)
    embedding2 = EMBEDDING_MODEL.encode(text2)
    return 1 - cosine(embedding1, embedding2)

# Process Dataset
def process_dataset(data, watermark_type="hard", delta=2.0, batch_size=32):
    """
    Process MS MARCO dataset and apply watermarking to each passage.
    """
    results = []
    total_batches = len(data) // batch_size + int(len(data) % batch_size != 0)

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch = data.iloc[batch_start:batch_end]

        print(f"Processing batch {batch_idx + 1}/{total_batches}...")
        for index, row in batch.iterrows():
            prompt = row["text"]
            try:
                watermarked_text = generate_text_with_watermark(prompt, watermark_type, delta)
                similarity = calculate_semantic_similarity(prompt, watermarked_text)
                results.append({
                    "id": row["id"],
                    "prompt": prompt,
                    "watermarked_text": watermarked_text,
                    "semantic_similarity": similarity
                })
            except Exception as e:
                print(f"Error processing ID {row['id']}: {e}")

    return results


# def process_dataset_with_centroid_and_distances(data, watermark_type="hard", delta=2.0, batch_size=1, num_results=5):
#     """
#     Process MS MARCO dataset, apply watermarking, compute distances from the centroid,
#     and calculate distances between the centroid and each watermarked output.
#     """
#     results = []
#     total_batches = len(data) // batch_size + int(len(data) % batch_size != 0)

#     for batch_idx in range(total_batches):
#         batch_start = batch_idx * batch_size
#         batch_end = min(batch_start + batch_size, len(data))
#         batch = data.iloc[batch_start:batch_end]

#         print(f"Processing batch {batch_idx + 1}/{total_batches}...")
#         for index, row in batch.iterrows():
#             prompt = row["text"]
#             try:
#                 # Generate multiple unwatermarked results and compute centroid
#                 multiple_results = generate_multiple_results(prompt, num_results)
#                 embeddings = compute_embeddings(multiple_results)
#                 centroid = compute_centroid(embeddings)

#                 # Generate multiple watermarked outputs
#                 watermarked_outputs = []
#                 watermarked_distances = []

#                 for _ in range(num_results):
#                     watermarked_text = generate_text_with_watermark(prompt, watermark_type, delta)
#                     watermarked_embedding = EMBEDDING_MODEL.encode([watermarked_text])[0]
#                     distance_from_centroid = compute_distance_from_centroid(watermarked_embedding, centroid)

#                     watermarked_outputs.append({
#                         "watermarked_text": watermarked_text,
#                         "distance_from_centroid": distance_from_centroid
#                     })
#                     watermarked_distances.append(distance_from_centroid)

#                 # Calculate semantic similarity for the first watermarked result (optional)
#                 similarity = calculate_semantic_similarity(prompt, watermarked_outputs[0]["watermarked_text"])

#                 # Store results
#                 results.append({
#                     "id": row["id"],
#                     "prompt": prompt,
#                     "centroid": centroid.tolist(),
#                     "watermarked_outputs": watermarked_outputs,
#                     "average_distance_from_centroid": np.mean(watermarked_distances),
#                     "semantic_similarity": similarity
#                 })

#             except Exception as e:
#                 print(f"Error processing ID {row['id']}: {e}")

#     return results

# Function to compute embeddings for a list of texts
def compute_embeddings(texts):
    """
    Compute embeddings for a list of texts.
    """
    return EMBEDDING_MODEL.encode(texts)

# Function to compute the centroid of the point cloud
def compute_centroid(embeddings):
    """
    Compute the centroid of the embedding space.
    """
    return np.mean(embeddings, axis=0)

# Function to compute the distance from the centroid
def compute_distance_from_centroid(embedding, centroid):
    """
    Compute the Euclidean distance from the centroid.
    """
    return euclidean(embedding, centroid)

# Save Results with Embedding
# def save_result_to_csv_with_embedding(result, file_path, mode="a"):
#     """
#     Save a single result to a CSV file in append mode, including embeddings.
#     """
#     fieldnames = [
#         "id",
#         "prompt",
#         "generated_text",
#         "distance_from_centroid",
#         "semantic_similarity",
#         "embedding"
#     ]
#     with open(file_path, mode=mode, newline="", encoding="utf-8") as file:
#         writer = csv.DictWriter(file, fieldnames=fieldnames)
#         if file.tell() == 0:  # Write header only if the file is empty
#             writer.writeheader()
#         writer.writerow(result)

# # Function to generate multiple results for the same prompt
# def generate_multiple_results(prompt, num_results=5):
#     """
#     Generate multiple results for the same prompt using the model.
#     """
#     results = []
#     for _ in range(num_results):
#         try:
#             result = generate_text_with_watermark(prompt, watermark_type=None)  # No watermark for centroid calculation
#             results.append(result)
#         except Exception as e:
#             print(f"Error generating result: {e}")
#     return results

# def process_non_watermarked_outputs(data, output_file, num_results=5):
#     """
#     Generate non-watermarked outputs and save each result to a CSV file, including embeddings.
#     """
#     print("Generating Non-Watermarked Outputs...")
#     for _, row in data.iterrows():
#         prompt = row["text"]
#         try:
#             # Generate multiple non-watermarked results
#             multiple_results = generate_multiple_results(prompt, num_results)
#             embeddings = compute_embeddings(multiple_results)
#             centroid = compute_centroid(embeddings)

#             for text, embedding in zip(multiple_results, embeddings):
#                 distance = compute_distance_from_centroid(embedding, centroid)
#                 result = {
#                     "id": row["id"],
#                     "prompt": prompt,
#                     "generated_text": text,
#                     "distance_from_centroid": distance,
#                     "semantic_similarity": calculate_semantic_similarity(prompt, text),
#                     "embedding": json.dumps(embedding.tolist())  # Convert embedding to JSON string
#                 }
#                 save_result_to_csv_with_embedding(result, output_file)
#         except Exception as e:
#             print(f"Error processing ID {row['id']}: {e}")

# def process_watermarked_outputs(data, watermark_type, output_file, delta=2.0, num_results=5):
#     """
#     Generate watermarked outputs (hard or soft) and save each result to a CSV file, including embeddings.
#     """
#     print(f"Generating {watermark_type.capitalize()} Watermarked Outputs...")
#     for _, row in data.iterrows():
#         prompt = row["text"]
#         try:
#             # Generate multiple watermarked results
#             multiple_results = []
#             for _ in range(num_results):
#                 text = generate_text_with_watermark(prompt, watermark_type=watermark_type, delta=delta)
#                 multiple_results.append(text)

#             embeddings = compute_embeddings(multiple_results)
#             centroid = compute_centroid(embeddings)

#             for text, embedding in zip(multiple_results, embeddings):
#                 distance = compute_distance_from_centroid(embedding, centroid)
#                 result = {
#                     "id": row["id"],
#                     "prompt": prompt,
#                     "generated_text": text,
#                     "distance_from_centroid": distance,
#                     "semantic_similarity": calculate_semantic_similarity(prompt, text),
#                     "embedding": json.dumps(embedding.tolist())  # Convert embedding to JSON string
#                 }
#                 save_result_to_csv_with_embedding(result, output_file)
#         except Exception as e:
#             print(f"Error processing ID {row['id']}: {e}")

# # Main Workflow
# if __name__ == "__main__":

#     # print("CUDA available:", torch.cuda.is_available())
#     # print("Current device:", torch.cuda.current_device())
#     # print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

#     # Load MS MARCO collection
#     collection_data = load_collection("collection.tsv")
#     print(f"Loaded {len(collection_data)} passages from MS MARCO.")
#     subset = collection_data.sample(20)

#     # File paths for outputs
#     non_watermarked_file = "non_watermarked_results.csv"
#     hard_watermarked_file = "hard_watermarked_results.csv"
#     soft_watermarked_file = "soft_watermarked_results.csv"

#     # Process and save non-watermarked results
#     process_non_watermarked_outputs(subset, non_watermarked_file)

#     # Process and save hard-watermarked results
#     process_watermarked_outputs(subset, "hard", hard_watermarked_file)

#     # Process and save soft-watermarked results
#     process_watermarked_outputs(subset, "soft", soft_watermarked_file, delta=DELTA)

#     print("Processing completed. Results saved to respective CSV files:")
#     print(f"- Non-Watermarked: {non_watermarked_file}")
#     print(f"- Hard-Watermarked: {hard_watermarked_file}")
#     print(f"- Soft-Watermarked: {soft_watermarked_file}")
    
    # # Process the dataset with hard and soft watermarking and compute distances
    # print("Processing with Hard Watermarking and Centroid Calculations...")
    # hard_watermarked_data = process_dataset_with_centroid_and_distances(subset, watermark_type="hard")
    # save_results_to_csv(hard_watermarked_data, "hard_watermarked_with_centroid.csv")
    # print("Saved Hard Watermarked Results with Centroid to 'hard_watermarked_with_centroid.csv'.")

    # print("Processing with Soft Watermarking and Centroid Calculations...")
    # soft_watermarked_data = process_dataset_with_centroid_and_distances(subset, watermark_type="soft", delta=DELTA)
    # save_results_to_csv(soft_watermarked_data, "soft_watermarked_with_centroid.csv")
    # print("Saved Soft Watermarked Results with Centroid to 'soft_watermarked_with_centroid.csv'.")

def generate_multiple_results(prompt, num_results=5, watermark_type=None, delta=2.0):
    results = []
    for _ in range(num_results):
        try:
            result = generate_text_with_watermark(prompt, watermark_type=watermark_type, delta=delta)
            results.append(result)
        except Exception as e:
            print(f"Error generating result: {e}")
    return results

def save_result_to_csv_with_embedding(result, file_path, mode="a"):
    """
    Save a single result to a CSV file in append mode, including embeddings.
    """
    fieldnames = [
        "id",
        "prompt",
        "generated_text",
        "distance_from_centroid",
        "semantic_similarity",
        "embedding"
    ]
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(result)


def process_outputs(data, output_file, watermark_type=None, delta=2.0, num_results=5):
    """
    Generalized function to process non-watermarked or watermarked outputs and save each result to a CSV file.
    """
    if watermark_type:
        print(f"Generating {watermark_type.capitalize()} Watermarked Outputs...")
    else:
        print("Generating Non-Watermarked Outputs...")
        
    for _, row in data.iterrows():
        prompt = row["text"]
        try:
            # Generate multiple results (watermarked or non-watermarked)
            multiple_results = generate_multiple_results(prompt, num_results, watermark_type, delta)
            embeddings = compute_embeddings(multiple_results)
            centroid = compute_centroid(embeddings)

            for text, embedding in zip(multiple_results, embeddings):
                distance = compute_distance_from_centroid(embedding, centroid)
                result = {
                    "id": row["id"],
                    "prompt": prompt,
                    "generated_text": text,
                    "distance_from_centroid": distance,
                    "semantic_similarity": calculate_semantic_similarity(prompt, text),
                    "embedding": json.dumps(embedding.tolist())  # Convert embedding to JSON string
                }
                save_result_to_csv_with_embedding(result, output_file)
        except Exception as e:
            print(f"Error processing ID {row['id']}: {e}")

# Main Workflow
if __name__ == "__main__":
    # Load MS MARCO collection
    collection_data = load_collection("collection.tsv")
    print(f"Loaded {len(collection_data)} passages from MS MARCO.")
    subset = collection_data.sample(20)

    # File paths for outputs
    non_watermarked_file = "non_watermarked_results.csv"
    hard_watermarked_file = "hard_watermarked_results.csv"
    soft_watermarked_file = "soft_watermarked_results.csv"

    # Process and save non-watermarked results
    process_outputs(subset, non_watermarked_file)

    # Process and save hard-watermarked results
    process_outputs(subset, hard_watermarked_file, watermark_type="hard")

    # Process and save soft-watermarked results
    process_outputs(subset, soft_watermarked_file, watermark_type="soft", delta=DELTA)

    print("Processing completed. Results saved to respective CSV files:")
    print(f"- Non-Watermarked: {non_watermarked_file}")
    print(f"- Hard-Watermarked: {hard_watermarked_file}")
    print(f"- Soft-Watermarked: {soft_watermarked_file}")
    