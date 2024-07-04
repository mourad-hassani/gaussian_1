from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

@dataclass
class GaussOutput(ModelOutput):
    mu: torch.FloatTensor = None
    std: torch.FloatTensor = None


class GaussModel(nn.Module):
    def __init__(self, model_name: str, gradient_checkpointing: bool = False) -> None:
        super().__init__()

        self.backbone: PreTrainedModel = AutoModel.from_pretrained(model_name)

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.hidden_size: int = self.backbone.config.hidden_size

        self.w_mu = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_var = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask, **_) -> GaussOutput:
        outputs: BaseModelOutput = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.
        emb = outputs.last_hidden_state[:, 0]

        mu = self.w_mu(emb)
        mu = self.activation(mu)

        log_var = self.w_var(emb)
        std = torch.sqrt(log_var.exp())

        return GaussOutput(mu=mu, std=std)