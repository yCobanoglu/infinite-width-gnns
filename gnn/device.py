import torch.cuda


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
