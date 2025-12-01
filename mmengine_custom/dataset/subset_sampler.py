import math
import torch
from torch.utils.data import Sampler
from mmengine.registry import DATA_SAMPLERS

"""
    Args:
        dataset (Dataset): The underlying dataset (MMDetection dataset).
        subset_ratio (float): Fraction of dataset to keep. 
            E.g., 0.1 = use 10% of images.
"""
@DATA_SAMPLERS.register_module()
class SubsetSampler(Sampler):
    def __init__(self,
                dataset,
                shuffle = True,
                subset_ratio=1.0,
                seed=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed

        num_samples = len(dataset)
        assert (0 < subset_ratio <= 1.0)

        subset_size = int(math.ceil(num_samples * subset_ratio))
        # pick first subset_size indices; we'll shuffle later if needed
        self.indices = list(range(num_samples))[:subset_size]
        self.num_samples = len(self.indices)


    def __iter__(self):
        # Optionally shuffle each epoch
        indices = self.indices
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
            indices = [self.indices[i] for i in indices]
        return iter(indices)

    def __len__(self):
        return self.num_samples    