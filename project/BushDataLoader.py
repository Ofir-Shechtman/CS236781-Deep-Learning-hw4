import os
import pathlib
import torch
import cs236781.download
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
torch.manual_seed(42)



class BushDataLoader(DataLoader):
    def __init__(self, batch_size, shuffle=True):
        DATA_DIR = pathlib.Path.home().joinpath('.pytorch-datasets')
        DATA_URL = 'http://vis-www.cs.umass.edu/lfw/lfw-bush.zip'

        _, dataset_dir = cs236781.download.download_data(out_path=DATA_DIR, url=DATA_URL, extract=True, force=False)

        im_size = 64
        tf = T.Compose([
            # Resize to constant spatial dimensions
            T.Resize((im_size, im_size)),
            # PIL.Image -> torch.Tensor
            T.ToTensor(),
            # Dynamic range [0,1] -> [-1, 1]
            T.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
        ])

        self.ds_gwb = ImageFolder(os.path.dirname(dataset_dir), tf)

        super().__init__(self.ds_gwb, batch_size, shuffle)

    @property
    def im_size(self):
        return self.ds_gwb[0][0].shape