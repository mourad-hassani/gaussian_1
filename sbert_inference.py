from sentence_transformers import SentenceTransformer, util
import torch
import json

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def print_colored(text, color):
    print(f"{color}{text}{Colors.RESET}")

model_save_path = "sentence_transformers/bert"

INPUT_FILE_PATH = "data/base_dataset/asymmetric_dataset.json"

model = SentenceTransformer(model_save_path)

first_sentences = []
second_sentences = []
ground_truth = []

with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    for element in data:
        first_sentences.append(element[0])
        second_sentences.append(element[1])
        ground_truth.append(element[2])

for first_sentence, second_sentence in zip(first_sentences, second_sentences):
    first_embedding = model.encode(first_sentence, convert_to_tensor=True)
    second_embedding = model.encode(second_sentence, convert_to_tensor=True)
    cosine_scores = util.cos_sim(first_embedding, second_embedding)[0]
    
    print(f"First sentence : {first_sentence}")
    print(f"Second sentence : {second_sentence}")

    print(f"Similarity : {cosine_scores.item()}")