from inference.inference import Inference

def run_inference():
    inference: Inference = Inference()
    
    output = inference.evaluate()

    outputs = {}

    for sent1, sent2, similarity, ground_truth in zip(output["sent1"], output["sent2"], output["similarity"], output["ground_truth"]):
        outputs[similarity] = {"sent1": sent1, "sent2": sent2}

    outputs = {key: outputs[key] for key in sorted(outputs)}

    for k, v in outputs.items():
        print(f"similarity: {k} => {v["sent1"]}, {v["sent2"]}")

if __name__ == "__main__":
    run_inference()