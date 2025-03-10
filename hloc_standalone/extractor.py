import torch
from pathlib import Path
import h5py
from types import SimpleNamespace
import cv2
import os
import random
import numpy as np
from loguru import logger
from tqdm import tqdm
from copy import deepcopy
from skimage.color import rgb2gray
from skimage.transform import resize
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from .third_party.SuperGluePretrainedNetwork.models.superpoint import SuperPoint as SP
from .netvlad.descriptor import NetVladExtractor
from .netvlad.descriptor import ImagesDataset as NetVladImageDataset
from .tools import map_tensor, interpolate_scan, h5autoclose


class SPImageDataset(Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
    }

    def __init__(self, root, conf):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        self.paths = []
        for g in conf.globs:
            self.paths += list(Path(root).glob('**/' + g))
        if len(self.paths) == 0:
            raise ValueError(f'Could not find any image in root: {root}.')
        self.paths = [i.relative_to(root) for i in self.paths]
        logger.info(f'Found {len(self.paths)} images in root {root}.')

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        image = cv2.imread(str(self.root / path), mode)
        if not self.conf.grayscale:
            image = image[:, :, ::-1]  # BGR to RGB
        if image is None:
            raise ValueError(f'Cannot read image {str(path)}.')
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        w, h = size

        if self.conf.resize_max and max(w, h) > self.conf.resize_max:
            scale = self.conf.resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            image = cv2.resize(
                image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': str(path),
            'image': image,
            'original_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.paths)


class HybridInMemImageDataset(Dataset):
    default_conf = {
        'local': {
            'grayscale': False,
            'resize_max': None,

        },
        'global':
            {
            },

    }

    @staticmethod
    def global_input_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __init__(self, images, local_conf, global_conf, name_format="{:06d}"):
        super().__init__()
        self.local_conf = SimpleNamespace(**{**self.default_conf['local'], **local_conf})
        self.global_conf = SimpleNamespace(**{**self.default_conf['global'], **global_conf})
        self.name_format = name_format
        self.images = images
        self.cached_global_transform = self.global_input_transform()

    def process_frame(self, img, frame_ind):
        frame_name = self.name_format.format(frame_ind)

        if img.dtype == np.uint8:
            local_img = img.astype(np.float32) / 255.
        else:
            local_img = img.copy()

        if self.local_conf.grayscale:
            local_img = rgb2gray(local_img)
        size = local_img.shape[:2][::-1]
        w, h = size

        if self.local_conf.resize_max and max(w, h) > self.local_conf.resize_max:
            scale = self.local_conf.resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            local_img = resize(local_img, (h_new, w_new), mode='reflect', order=1)

        if self.local_conf.grayscale:
            local_img = local_img[None]
        else:
            local_img = local_img.transpose((2, 0, 1))  # HxWxC to CxHxW
        local_data = {
            'index': frame_ind,
            'name': frame_name,
            'image': local_img,
            'original_size': np.array(size),
        }
        # Prepairing global image
        global_img = self.cached_global_transform(img)
        global_data = {'data': global_img, 'name': frame_name}
        return local_data, global_data

    def __getitem__(self, idx):
        image = self.images[idx]
        return self.process_frame(image, idx)

    def __len__(self):
        return len(self.images)


class SuperPointNetVladExtractor:
    default_config = {
        'local': {
            'model': {
                'name': 'superpoint',
                'nms_radius': 3,
                'max_keypoints': 4096,
                'keypoint_threshold': 0.005,
                'remove_borders': 4
            },
            'preprocessing': {
                'grayscale': True,
                'resize_max': 1024,
            },
            'loader': {
                'workers': 8,
                'batch_size': 4
            },
            'storage': {
                'as_half': False,
            }
        },
        'global': {
            'model': {},
            'loader': {
                'workers': 8,
                'batch_size': 4
            }
        }
    }

    def __init__(self, device=None, extractor_config=None, show_progress=True):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        config = deepcopy(self.default_config)
        if extractor_config is not None:
            config.update(extractor_config)
        self.config = config
        self.local_extractor = SP(config['local']['model']).eval().to(device)
        self.global_extractor = NetVladExtractor(device=device, **config['global']['model'])
        self.show_progress = show_progress

    def write_database(self, database_path, image_dir, vertex_mapping_dir=None):
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        desc_file = h5py.File(str(database_path), 'w')
        self.extract(desc_file, image_dir, vertex_mapping_dir=vertex_mapping_dir)
        desc_file.close()

    def append_database(self, database_path, image_dir, vertex_mapping_dir=None):
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        desc_file = h5py.File(str(database_path), 'a')
        self.extract(desc_file, image_dir, vertex_mapping_dir=vertex_mapping_dir)
        desc_file.close()

    def inmem_database(self, image_dir, cache_path=None, vertex_mapping_dir=None):
        if cache_path is None:
            desc_file = h5py.File('tmp.' + os.path.basename(image_dir) + '.{}.h5'.format(random.randint(0, 65536)), 'w', driver='core',
                                  backing_store=False)
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            desc_file = h5py.File(cache_path, 'w', driver='core')
        desc_file = h5autoclose(desc_file)
        self.extract(desc_file, image_dir, vertex_mapping_dir=vertex_mapping_dir)
        return desc_file

    def extract_from_frames_hybrid(self, desc_file, images, xyz_maps = None):
        logger.info("Extracting features...")
        local_config = self.config['local']['preprocessing'] if 'preprocessing' in self.config['local'] else {}
        global_config = self.config['global']['preprocessing'] if 'preprocessing' in self.config['global'] else {}
        dataset = HybridInMemImageDataset(images, local_config, global_config)
        loader_params = self.config['local']['loader']
        loader = DataLoader(dataset, num_workers=loader_params['workers'],
                            batch_size=loader_params['batch_size'])
        with torch.no_grad():
            progress_func = tqdm if self.show_progress else (lambda x: x)
            for local_data, global_data in progress_func(loader):
                batch_pred = self.local_extractor(map_tensor(local_data, lambda x: x.to(self.device)))
                for ind in range(local_data['image'].size(0)):
                    pred = {k: v[ind].cpu().numpy() for k, v in batch_pred.items()}

                    pred['image_size'] = original_size = local_data['original_size'][ind].numpy()
                    if 'keypoints' in pred:
                        size = np.array(local_data['image'].shape[-2:][::-1])
                        scales = (original_size / size).astype(np.float32)
                        pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

                    if xyz_maps is not None:
                        scan = xyz_maps[local_data['index'][ind]]
                        keypoints3d, valid3d = interpolate_scan(scan, pred['keypoints'])
                        keypoints3d[~valid3d] = float('nan')
                        pred['keypoints3d'] = keypoints3d

                    if self.config['local']['storage']['as_half']:
                        for k in pred:
                            dt = pred[k].dtype
                            if (dt == np.float32) and (dt != np.float16):
                                pred[k] = pred[k].astype(np.float16)

                    grp = desc_file.create_group(local_data['name'][ind])
                    for k, v in pred.items():
                        grp.create_dataset(k, data=v)

                names = global_data['name']
                data = global_data['data'].to(self.device)
                image_encoding = self.global_extractor.model.encoder(data)
                vlad_encoding = self.global_extractor.model.pool(image_encoding)
                for imname, desc in zip(names, vlad_encoding):
                    grp = desc_file[imname]
                    grp.create_dataset('global_descriptor', data=desc.cpu().numpy())

    def extract(self, desc_file, image_dir, vertex_mapping_dir=None):
        logger.info("Extracting local features...")
        dataset = SPImageDataset(image_dir, self.config['local']['preprocessing'])
        loader_params = self.config['local']['loader']
        loader = DataLoader(dataset, num_workers=loader_params['workers'],
                            batch_size=loader_params['batch_size'])
        with torch.no_grad():
            progress_func = tqdm if self.show_progress else (lambda x: x)
            for data in progress_func(loader):
                batch_pred = self.local_extractor(map_tensor(data, lambda x: x.to(self.device)))
                for ind in range(data['image'].size(0)):
                    pred = {k: v[ind].cpu().numpy() for k, v in batch_pred.items()}

                    pred['image_size'] = original_size = data['original_size'][ind].numpy()
                    if 'keypoints' in pred:
                        size = np.array(data['image'].shape[-2:][::-1])
                        scales = (original_size / size).astype(np.float32)
                        pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

                    if vertex_mapping_dir is not None:
                        scan = np.load(os.path.join(vertex_mapping_dir,
                                                    os.path.splitext(data['name'][ind])[0] + '.xyz.npz'))['xyz']
                        keypoints3d, valid3d = interpolate_scan(scan, pred['keypoints'])
                        keypoints3d[~valid3d] = float('nan')
                        pred['keypoints3d'] = keypoints3d

                    if self.config['local']['storage']['as_half']:
                        for k in pred:
                            dt = pred[k].dtype
                            if (dt == np.float32) and (dt != np.float16):
                                pred[k] = pred[k].astype(np.float16)

                    grp = desc_file.create_group(data['name'][ind])
                    for k, v in pred.items():
                        grp.create_dataset(k, data=v)

                del pred
        logger.info("Extracting global features...")
        dataset = NetVladImageDataset(dataset.paths, root_folder=dataset.root)
        loader_params = self.config['global']['loader']
        loader = DataLoader(dataset, num_workers=loader_params['workers'],
                            batch_size=loader_params['batch_size'])
        with torch.no_grad():
            for batch in progress_func(loader):
                names = batch['name']
                data = batch['data'].to(self.device)
                image_encoding = self.global_extractor.model.encoder(data)
                vlad_encoding = self.global_extractor.model.pool(image_encoding)
                for imname, desc in zip(names, vlad_encoding):
                    grp = desc_file[imname]
                    grp.create_dataset('global_descriptor', data=desc.cpu().numpy())
        return desc_file
