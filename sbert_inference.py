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

model_save_path = "sentence_transformers/all_mpnet"

INPUT_FILE_PATH = "data/base_dataset/small_dataset.json"

model = SentenceTransformer(model_save_path)

corpus = []
ground_truth = []

with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)[:5]
    for element in data:
        corpus.append(element[1])
        ground_truth.append(element[2])

corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

queries = [data[0][0]]

top_k = min(5, len(corpus))

for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    print(f"Query: {queries[0]}")
    
    for score, idx in zip(top_results[0], top_results[1]):
        score_text = "(Score: {:.4f})".format(score)
        print(corpus[idx], f" {score_text}")
        print(ground_truth[idx])
        print_colored(text=corpus[idx].split("[SEP]")[0].replace("[CLS]", "==============>"), color=Colors.BLUE if score > 0.9 else Colors.RED)