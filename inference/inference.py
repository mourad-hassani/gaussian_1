import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from transformers import AutoTokenizer

from gauss_model import GaussModel, GaussOutput
from parameters import MODEL_NAME, INFERENCE_DEVICE, BATCH_SIZE, NUM_WORKERS, MAX_SEQ_LEN, INPUT_FILE_PATH, SPECIAL_TOKENS, TEMPERATURE
from utils.similarity import asymmetrical_kl_sim

class Inference:
    def __init__(self):
        self.model = GaussModel(MODEL_NAME, True).eval().to(INFERENCE_DEVICE)
        self.model.load_state_dict(torch.load('temporal_bert.pth', map_location=torch.device(INFERENCE_DEVICE)))

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length = MAX_SEQ_LEN, use_fast = False)
        
        self.sentences1 = ["[CLS] these next 25 months [SEP] 18 october 2023 [SEP]", "[CLS] tomorrow [SEP] 29 june 2023 [SEP]", "[CLS] yesterday [SEP] 01 july 2023 [SEP]", "[CLS] 12 september 2023 [SEP] 11 july 2023 [SEP]"]
        self.sentences2 = ["[CLS] from 01 november 2025 to 30 november 2027 [SEP] 15 june 2024 [SEP]"] * len(self.sentences1)
        self.scores = [1.0] * len(self.sentences1)

    def tokenize(self, batch: list[str]) -> BatchEncoding:
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=MAX_SEQ_LEN, add_special_tokens=SPECIAL_TOKENS)
    
    def data_loader(self, sentences: list[str]):
        return DataLoader(sentences, collate_fn=self.tokenize, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    @torch.inference_mode()
    def sim_fn(self, sent1: str, sent2: str) -> float:
            sent1: GaussOutput = self.encode_fn(sent1)
            sent2: GaussOutput = self.encode_fn(sent2)
            return asymmetrical_kl_sim(sent1.mu, sent1.std, sent2.mu, sent2.std).item() / TEMPERATURE

    @torch.inference_mode()
    def encode_fn(self, sentence: str, **_) -> GaussOutput:
        self.model.eval()

        output: GaussOutput = None

        for batch in self.data_loader([sentence]):
            output = self.model.forward(**batch.to(INFERENCE_DEVICE))
            break

        return output
    
    @torch.inference_mode()
    def evaluate(self) -> dict:
        similarities: list[float] = []
        
        for sent1, sent2 in zip(self.sentences1, self.sentences2):
            similarities.append(self.sim_fn(sent1, sent2))
        
        return {"sent1": self.sentences1, "sent2": self.sentences2, "similarity": similarities, "ground_truth": self.scores}