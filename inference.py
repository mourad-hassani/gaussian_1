from inference.inference import Inference

def run_inference():
    inference: Inference = Inference()
    
    output = inference.evaluate()

    for sent1, sent2, similarity, ground_truth in zip(output["sent1"], output["sent2"], output["similarity"], output["ground_truth"]):
        print(f"{sent1}, {sent2}, {similarity}, {ground_truth}")

if __name__ == "__main__":
    run_inference()