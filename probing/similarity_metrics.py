import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity



def angular_distance(
    x1: Tensor,
    x2: Tensor,
    dim: int = 1,
    eps: float = 1e-8
) -> Tensor:
    cos_sim = F.cosine_similarity(x1, x2, dim, eps)
    res = torch.arccos(cos_sim)
    return res / torch.pi
