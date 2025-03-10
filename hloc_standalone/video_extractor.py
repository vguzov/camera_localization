import torch
from pathlib import Path
import h5py
from types import SimpleNamespace
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader, IterableDataset
from .third_party.SuperGluePretrainedNetwork.models.superpoint import SuperPoint as SP
from .netvlad.descriptor import NetVladExtractor
from .tools import map_tensor, interpolate_scan, h5autoclose
from videoio import VideoReader, read_video_params
from torchvision import transforms
from skimage.color import rgb2gray
from skimage.transform import resize
import random

class HybridVideoDataset(IterableDataset):
    default_conf = {
        'local':{
            'grayscale': False,
            'resize_max': None,
        },
        'global':
            {
            }
    }

    @staticmethod
    def global_input_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __init__(self, videopath, local_conf, global_conf, frame_step = 1, start_frame=0, end_frame=None, video_resolution = None, remember_original_scale = True):
        super().__init__()
        self.local_conf = SimpleNamespace(**{**self.default_conf['local'], **local_conf})
        self.global_conf = SimpleNamespace(**{**self.default_conf['global'], **global_conf})
        self.videopath = videopath
        self.frame_step = frame_step
        self.start_frame = start_frame
        self.end_frame = end_frame
        videoparams = read_video_params(videopath)
        self.seqlen = videoparams['length'] if 'length' in videoparams else 0
        if self.end_frame is not None:
            self.seqlen = min(self.seqlen, self.end_frame)
        self.seqlen = (self.seqlen-start_frame)//frame_step
        original_resolution = [videoparams["width"], videoparams["height"]]
        self.cached_global_transform = self.global_input_transform()
        self.override_video_resolution = video_resolution
        if video_resolution is not None and not remember_original_scale:
            self.resolution = np.array(video_resolution)
        else:
            self.resolution = np.array(original_resolution)

    def process_frame(self, img, frame_ind):
        frame_name = f"{frame_ind:06d}"
        # Preparing data for local feature extraction
        if self.local_conf.grayscale:
            local_img = rgb2gray(img.copy()).astype(np.float32)
        else:
            local_img = img.copy()
            local_img = local_img.astype(np.float32)/255.
        size = local_img.shape[:2][::-1]
        w, h = size

        if self.local_conf.resize_max and max(w, h) > self.local_conf.resize_max:
            scale = self.local_conf.resize_max / max(h, w)
            h_new, w_new = int(round(h*scale)), int(round(w*scale))
            local_img = resize(local_img, (h_new, w_new), mode='reflect', order=1)

        if self.local_conf.grayscale:
            local_img = local_img[None]
        else:
            local_img = local_img.transpose((2, 0, 1))  # HxWxC to CxHxW
        local_data = {
                'name': frame_name,
                'image': local_img,
                'original_size': self.resolution,
        }
        # Preparing data for global feature extraction
        global_img = self.cached_global_transform(img)
        global_data = {'data':global_img, 'name':frame_name}
        return local_data, global_data

    def __iter__(self):
        if self.override_video_resolution is None:
            self.videoreader = VideoReader(self.videopath)
        else:
            self.videoreader = VideoReader(self.videopath, output_resolution=self.override_video_resolution)
        self.videoiter = iter(self.videoreader)
        self.frame_ind = -1
        self.frame_n = 1 + self.start_frame
        return self

    def __len__(self):
        return self.seqlen

    def __next__(self):
        frame = None
        while self.frame_n > 0:
            frame = next(self.videoiter)
            self.frame_n -= 1
            self.frame_ind += 1
        self.frame_n = self.frame_step
        if (self.end_frame is not None) and (self.frame_ind >= self.end_frame):
            raise StopIteration
        return self.process_frame(frame, self.frame_ind)


class SuperPointNetVladVideoExtractor:
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

            'storage': {
                'as_half': False,
            }
        },
        'global': {
            'model': {},
            'preprocessing': {}

        },
        'loader': {
            'batch_size': 4
        }
    }

    def __init__(self, device=None, extractor_config=None, video_resolution=None, show_progress=True):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        config = deepcopy(self.default_config)
        if extractor_config is not None:
            config.update(extractor_config)
        self.config = config
        self.local_extractor = SP(config['local']['model']).eval().to(device)
        self.global_extractor = NetVladExtractor(device=device, **config['global']['model'])
        self.override_video_resolution = video_resolution
        self.show_progress = show_progress

    def write_database(self, database_path, videopath, vertex_mapping_dir=None, frame_step = 1, start_frame = 0, end_frame=None):
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        desc_file = h5py.File(str(database_path), 'w')
        self.extract(desc_file, videopath, vertex_mapping_dir=vertex_mapping_dir, frame_step=frame_step, start_frame=start_frame, end_frame=end_frame)
        desc_file.close()

    def append_database(self, database_path, videopath, vertex_mapping_dir=None, frame_step = 1, start_frame = 0, end_frame=None):
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        desc_file = h5py.File(str(database_path), 'a')
        self.extract(desc_file, videopath, vertex_mapping_dir=vertex_mapping_dir, frame_step=frame_step, start_frame=start_frame, end_frame=end_frame)
        desc_file.close()

    def inmem_database(self, videopath, cache_path = None, vertex_mapping_dir=None, frame_step = 1, start_frame = 0, end_frame=None):
        if cache_path is None:
            desc_file = h5py.File('tmp.'+os.path.basename(videopath)+'.{}.h5'.format(random.randint(0,65536)), 'w', driver='core', backing_store=False)
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            desc_file = h5py.File(cache_path, 'w', driver='core')
        desc_file = h5autoclose(desc_file)
        self.extract(desc_file, videopath, vertex_mapping_dir=vertex_mapping_dir, frame_step=frame_step, start_frame=start_frame, end_frame=end_frame)
        return desc_file

    def extract(self, desc_file, videopath, vertex_mapping_dir=None, frame_step = 1, start_frame = 0, end_frame=None):
        dataset = HybridVideoDataset(videopath, self.config['local']['preprocessing'],
                                     self.config['global']['preprocessing'],
                                     frame_step=frame_step, start_frame=start_frame, end_frame=end_frame,
                                     video_resolution=self.override_video_resolution)
        loader_params = self.config['loader']
        loader = DataLoader(dataset, num_workers=1, batch_size=loader_params['batch_size'])
        with torch.no_grad():
            progress_func = tqdm if self.show_progress else (lambda x: x)
            for local_data, global_data in progress_func(loader):
                data = local_data
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

                batch = global_data
                names = batch['name']
                data = batch['data'].to(self.device)
                image_encoding = self.global_extractor.model.encoder(data)
                vlad_encoding = self.global_extractor.model.pool(image_encoding)
                for imname, desc in zip(names, vlad_encoding):
                    grp = desc_file[imname]
                    grp.create_dataset('global_descriptor', data=desc.cpu().numpy())
        return desc_file






