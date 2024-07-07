from gauss_model import GaussOutput
import torch
from torch import distributions

def asymmetrical_kl_sim(mu1: torch.FloatTensor, std1: torch.FloatTensor, mu2: torch.FloatTensor, std2: torch.FloatTensor) -> torch.Tensor:
    """
    Computes the KL similarity between two normal distributions and returns a tensor with shape (batch_size)
    """
    
    lower_bound1: torch.FloatTensor = mu1 - (3 * std1)
    upper_bound1: torch.FloatTensor = mu1 + (3 * std1)
    
    lower_bound2: torch.FloatTensor = mu2 - (3 * std2)
    upper_bound2: torch.FloatTensor = mu2 + (3 * std2)

    is_in: bool = False
    if (lower_bound2 <= lower_bound1).all() and (upper_bound1 <= upper_bound2).all():
        is_in = True

    p1 = distributions.normal.Normal(mu1, std1)
    p2 = distributions.normal.Normal(mu2, std2)

    sim = torch.nn.Tanh(distributions.kl.kl_divergence(p1, p2))

    if is_in:
        return -sim.mean(dim=-1)
    
    return sim.mean(dim=-1)