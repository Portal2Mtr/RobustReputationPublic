"""Dataset base class

Base class for all dataset classes

"""
from torch.utils.data import Dataset


class DatasetBase(Dataset):

    def __init__(self):
        super().__init__()
        self.num_features = None
        self.num_classes = None
        self.downsample = False
        self.overwrite_synth_labels = False
        return
