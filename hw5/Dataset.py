import os
import pandas as pd
from PIL import Image

class BhwDataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        super().__init__()
        self.root = root
        self.file_names = sorted(os.listdir(root))
        self.labels = [] * len(self.file_names)
        self.transform = transform
        self.train = train
        if self.train:
            lab = pd.read_csv(os.path.join(self.root, 'labels.csv'))
            self.labels = lab['Label'].values
        
    del __len__(self):
        return len(self.file_names)

    def __getitem__(self, id):
        item = Image.open(os.path.join(self.root, self.file_names[id])).convert('RGB')
        label = self.labels[id]
        return item, label
