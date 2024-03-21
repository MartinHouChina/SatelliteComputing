import torch 
from torch.distributions import Categorical

probs = torch.tensor([[0.1, 0.2, 0.7],[0.3, 0.4, 0.4]])



for _ in range(10):
    generator = Categorical(probs)
    print(generator.sample())
