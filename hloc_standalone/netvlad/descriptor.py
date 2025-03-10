# Based on https://github.com/Nanne/pytorch-NetVlad

import os
import torch
import torchvision.models
from torchvision import transforms
import torch.nn as nn
import h5py
from torch.utils.data import Dataset, DataLoader
from .netvlad import NetVLAD
from PIL import Image
from tqdm import tqdm
from loguru import logger

class NetVladExtractor:
    netvlad_default_params = dict(num_clusters=64, dim=512,
                 normalize_input=True, vladv2=False)
    def __init__(self, weights_path='weights/checkpoint.pth.tar', device = None, **kwargs):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        netvlad_params = {}
        for k,v in self.netvlad_default_params.items():
            if k in kwargs:
                netvlad_params[k] = kwargs[k]
            else:
                netvlad_params[k] = v
        weight_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), weights_path)

        #Creating encoder
        encoder = torchvision.models.vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]

        # if using pretrained then only train conv5_1, conv5_2, and conv5_3
        for l in layers[:-5]:
            for p in l.parameters():
                p.requires_grad = False

        self.encoder = nn.Sequential(*layers)
        model = nn.Module()
        model.add_module('encoder', self.encoder)

        # Creating NetVLAD layer
        self.netvlad_layer = NetVLAD(**netvlad_params)
        model.add_module('pool', self.netvlad_layer)
        checkpoint = torch.load(weight_full_path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model.to(device)
        logger.debug("=> loaded NetVLAD checkpoint (epoch {})"
              .format(checkpoint['epoch']))

    def extract(self, images, batch_size = 4):
        transform = ImagesDataset.input_transform()

        with torch.no_grad():
            nn_inputs = torch.stack([transform(img) for img in images], dim=0)
            nn_inputs = nn_inputs.to(self.device)
            descs = []
            for nn_input in torch.split(nn_inputs, batch_size):
                image_encoding = self.model.encoder(nn_input)
                vlad_encoding = self.model.pool(image_encoding)
                descs.append(vlad_encoding)
            descs = torch.cat(descs, dim=0)
        return descs

    def create_database_old(self, database_path, image_paths):
        desc_file = h5py.File(str(database_path), 'a')
        descs = self.extract([Image.open(x) for x in image_paths])
        for desc, imname in zip(descs, image_paths):
            grp = desc_file.create_group(imname)
            grp.create_dataset('global_descriptor', data=desc.cpu().numpy())

    def create_database(self, database_path, image_paths, batch_size=4):
        desc_file = h5py.File(str(database_path), 'a')
        dataset = ImagesDataset(image_paths)
        dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                names = batch['name']
                data = batch['data'].to(self.device)
                image_encoding = self.model.encoder(data)
                vlad_encoding = self.model.pool(image_encoding)
                for imname, desc in zip(names, vlad_encoding):
                    grp = desc_file.create_group(imname)
                    grp.create_dataset('global_descriptor', data=desc.cpu().numpy())
        desc_file.close()



class ImagesDataset(Dataset):
    @staticmethod
    def input_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __init__(self, images_paths, root_folder = None):
        super().__init__()
        self.images_names = list(images_paths)
        if root_folder is not None:
            self.images_paths = [os.path.join(root_folder, x) for x in self.images_names]
        else:
            self.images_paths = self.images_names
        self.cached_input_transform = self.input_transform()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        imgpath = str(self.images_paths[item])
        img = Image.open(imgpath)
        return {'data':self.cached_input_transform(img), 'name':str(self.images_names[item])}

