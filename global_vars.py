import torch 

global DEVICE
DEVICE = 'cuda:4' if torch.cuda.is_available() else 'cpu:4'
