import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from transformers import AutoTokenizer

from gauss_model import GaussModel, GaussOutput
from parameters import MODEL_NAME, INFERENCE_DEVICE, BATCH_SIZE, NUM_WORKERS, MAX_SEQ_LEN, INPUT_FILE_PATH, SPECIAL_TOKENS
from utils.similarity import asymmetrical_kl_sim

class Inference:
    def __init__(self):
        self.model = GaussModel(MODEL_NAME, True).eval().to(INFERENCE_DEVICE)
        self.model.load_state_dict(torch.load('temporal_bert.pth', map_location=torch.device(INFERENCE_DEVICE)))

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length = MAX_SEQ_LEN, use_fast = False)
                
        self.sentences2 = [
            "[CLS] 12 june 2023 [SEP] 18 october 2023 [SEP]",
            "[CLS] 20 june 2023 [SEP] 18 october 2023 [SEP]",
            "[CLS] 01 june 2023 [SEP] 18 october 2023 [SEP]",
            "[CLS] 16 june 2023 [SEP] 18 october 2023 [SEP]",
            "[CLS] 15 june 2023 [SEP] 29 june 2023 [SEP]",
            "[CLS] from 10 june 2023 to 15 june 2023 [SEP] 01 july 2023 [SEP]",
            "[CLS] from 10 june 2024 to 15 june 2024 [SEP] 15 june 2023 [SEP]"
        ]
        
        self.sentences1 = ["[CLS] What drugs did the patient take these past 5 days? [SEP] 15 june 2023 [SEP]"] * len(self.sentences2)
        self.scores = [1.0] * len(self.sentences1)

    def tokenize(self, batch: list[str]) -> BatchEncoding:
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=MAX_SEQ_LEN, add_special_tokens=SPECIAL_TOKENS)
    
    def data_loader(self, sentences: list[str]):
        return DataLoader(sentences, collate_fn=self.tokenize, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    def sim_fn(self, sent1: str, sent2: str) -> float:
            sent1: GaussOutput = self.encode_fn(sent1)
            sent2: GaussOutput = self.encode_fn(sent2)
            return asymmetrical_kl_sim(sent1.mu, sent1.std, sent2.mu, sent2.std).item()

    @torch.inference_mode()
    def encode_fn(self, sentence: str, **_) -> GaussOutput:
        self.model.eval()

        output: GaussOutput = None

        for batch in self.data_loader([sentence]):
            output = self.model.forward(**batch.to(INFERENCE_DEVICE))
            break

        return output
    
    def evaluate(self) -> dict:
        similarities: list[float] = []
        
        for sent1, sent2 in zip(self.sentences1, self.sentences2):
            similarities.append(self.sim_fn(sent1, sent2))
        
        return {"sent1": self.sentences1, "sent2": self.sentences2, "similarity": similarities, "ground_truth": self.scores}