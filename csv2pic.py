from easydl.common.wheel import join_path
from config import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
import torch
import sys

sys.path[0] = '/home/lab-wu.shibin/SFDA'

class BaseImageDataset(Dataset):
    def __init__(self, transform=None, return_id=False):
        self.return_id = return_id
        self.transform = transform or (lambda x : x)
        self.datas = []
        self.labels = []
        self.idx = []

    def __getitem__(self, index):
        data = pd.read_csv(self.datas[index],encoding='UTF-8')
        img = data[['L1','L2','L3','L4','L5','L6','L7','L8','R1','R2','R3','R4','R5','R6','R7','R8']]
        img_norm = img.transpose()
        img_as_np = np.asarray(img_norm).astype(float)
        img_as_img = Image.fromarray(img_as_np) 
        im = img_as_img.convert('L')
        im = self.transform(im)
        if not self.return_id:
            return im, self.labels[index], self.idx[index]
        return im, self.labels[index], self.idx[index], index

    def __len__(self):
        return len(self.datas)

class DatasetFromCSV(BaseImageDataset):
    def __init__(self, list_path, path_prefix='', transform=None, return_id=False, num_classes=None, filter=None):
        super(DatasetFromCSV, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            
            data = [[line.split()[0], line.split()[1], line.split()[2] if len(line.split()) > 1 else '0'] for line in f.readlines() if
                    line.strip()]
            
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            
            try:
                self.idx = [int(x[2]) for x in data]
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e
        ans = [(x, y, z) for (x, y, z) in zip(self.datas, self.labels, self.idx) if filter(y)]
        self.datas, self.labels, self.idx = zip(*ans)

        self.num_classes = num_classes or max(self.labels) + 1

train_transform = Compose([ #Tansform setting: resnet-50-resize224; vggnet-16-resize256-crop224
    Resize(256,interpolation=Image.LANCZOS),
    CenterCrop(224),
    ToTensor()
])
test_transform = Compose([
    Resize(256,interpolation=Image.LANCZOS),
    CenterCrop(224),
    ToTensor()
])

source_train_ds = DatasetFromCSV(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                            transform=train_transform#, filter=(lambda x: x in source_classes)
                            )
train_size = int(len(source_train_ds) * 0.8)
val_size = len(source_train_ds) - train_size

source_train_ds, source_test_ds = torch.utils.data.random_split(source_train_ds, [len(source_train_ds)-600, 600]) #1400

target_train_ds = DatasetFromCSV(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=train_transform#, filter=(lambda x: x in target_classes)
                            )
target_test_ds = DatasetFromCSV(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=test_transform#, filter=(lambda x: x in target_classes)
                            )
source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                             #sampler=sampler, 
                             num_workers=args.data.dataloader.data_workers, drop_last=True)
source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=True,
                             num_workers=1, drop_last=False)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                             num_workers=args.data.dataloader.data_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=True,
                             num_workers=1, drop_last=True)
