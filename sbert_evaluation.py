from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import json
from pathlib import Path

dataset_path: Path = Path("data/base_dataset")
dataset_file_name = "base_dataset_close.json"

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

model_name = "sentence-transformers/all-mpnet-base-v2"
train_batch_size = 16
num_epochs = 1
model_save_path = ("output/finetune-" + model_name + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

model = SentenceTransformer(model_name)

logging.info("Read train dataset")

train_samples = []
dev_samples = []
test_samples = []

with open(os.path.join(dataset_path, dataset_file_name)) as f:
    data = json.load(f)[:1000000]
    for i, row in enumerate(data):
        inp_example = InputExample(texts=[row[0], row[1]], label=float(row[2]))
        if i < (0.9 * len(data)):
            train_samples.append(inp_example)
        elif i < (0.95 * len(data)):
            dev_samples.append(inp_example)
        else:
            test_samples.append(inp_example)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CoSENTLoss(model=model)


logging.info("Read dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="sts-dev")

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
logging.info("Warmup-steps: {}".format(warmup_steps))

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
)

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")
test_evaluator(model, output_path=model_save_path)