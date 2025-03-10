from argparse import ArgumentParser
from hloc_standalone.extractor import SuperPointNetVladExtractor
from hloc_standalone.video_extractor import SuperPointNetVladVideoExtractor
from hloc_standalone.matcher import SuperGluePrefilteringMatcher
from hloc_standalone.localizer import Localizer
from hloc_standalone.tools import h5autoclose
import h5py
import json
import os
import torch
from pathlib import Path
from videoio import read_video_params
from skimage.io import imread
from loguru import logger

colmap_models = {
    'opencv': 'OPENCV',
    'perspective': 'SIMPLE_PINHOLE',
    'opencv_fisheye': 'OPENCV_FISHEYE'
}

# Add known cameras here for convenience
known_cameras = {
    'example_camera1': {'camera_model': 'opencv',
                        'camera_params': [871.80021916157159, 885.57997989678529, 961.54424567006754, 550.68646879951450,
                                          -0.25462584991295, 0.08039095012756, 0.00014583290360, -0.00001397345667]},
    'example_camera2': {'camera_model': 'opencv',
                        'camera_params': [870.39802320390481, 883.77772671373339, 977.24123059216777, 550.42154564060434,
                                          -0.26512391289441, 0.09675221734815, 0.00000604206872, 0.00012467424427]},
}

parser = ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Input frames: images folder or video file")
parser.add_argument("-db", "--db_features", required=True, help="Database of extracted features from anchor images (h5)")
parser.add_argument("-o", "--outfile", required=True, help="Output localization results (json)")

parser.add_argument("-f", "--filecache_prefix", help="Path for caching intermediate results (query features, matches). "
                                                     "If not provided, in-memory processing is used")
parser.add_argument("-cm", "--camera_model", default="opencv", choices=['perspective', 'opencv', 'opencv_fisheye'] + list(known_cameras.keys()))
parser.add_argument("-cp", "--camera_params", nargs='*', type=float, help="Camera parameters in OpenCV format: fx fy cx cy k1 k2 p1 p2. "
                                                                          "If camera_model is from known_cameras, this argument is ignored.")
parser.add_argument("-bs", "--batch_size", default=4, type=int)
parser.add_argument("-n", "--frame_step", default=1, type=int, help="Process every n-th frame")
parser.add_argument("-s", "--start_frame", default=0, type=int, help="Start processing from particular frame number")
parser.add_argument("-e", "--end_frame", default=None, type=int, help="End processing at particular frame number")
parser.add_argument("-sgw", "--sg_weights", choices=['indoor', 'outdoor'], default='outdoor', help="SuperGlue weights")
parser.add_argument("-pres", "--processing_resolution", nargs=2, type=int, default=None,
                    help="Resize input frames to particular resolution (width height)")
parser.add_argument("-rme", "--ransac_max_error", type=float, default=48.,
                    help="RANSAC max reprojection error (in pixels), errors above are considered failures")

args = parser.parse_args()

if args.camera_model in known_cameras:
    camera = known_cameras[args.camera_model]
    args.camera_model = camera['camera_model']
    args.camera_params = camera['camera_params']

camera_params = args.camera_params

matcher_config = {
    'matcher': {
        'model': {
            'name': 'superglue',
            'weights': args.sg_weights,
            'sinkhorn_iterations': 50,
        }
    },
    'filter': {
        'matches_per_query': 40
    }
}

db_features = h5py.File(args.db_features, "r")

logger.info("Extraction stage...")
if os.path.isdir(args.input):
    assert args.processing_resolution is None, "Picture scaling is not supported"
    extractor_config = {
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
                'batch_size': args.batch_size
            },
            'storage': {
                'as_half': False,
            }
        },
        'global': {
            'model': {},
            'loader': {
                'workers': 8,
                'batch_size': args.batch_size
            }
        }
    }
    extractor = SuperPointNetVladExtractor(extractor_config=extractor_config)
    if args.filecache_prefix:
        query_fname = args.filecache_prefix + '.query.h5'
        if os.path.isfile(query_fname):
            os.remove(query_fname)
        extractor.write_database(query_fname, args.input)
        query_features = h5autoclose(h5py.File(query_fname, 'r'))
    else:
        query_features = extractor.inmem_database(args.input)
    paths = []
    for g in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
        paths += list(Path(args.input).glob('**/' + g))
    resolution = imread(paths[0]).shape[:2][::-1]
else:
    extractor_config = {
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
            'batch_size': args.batch_size
        }
    }
    videoextractor = SuperPointNetVladVideoExtractor(extractor_config=extractor_config, video_resolution=args.processing_resolution)
    if args.filecache_prefix:
        query_fname = args.filecache_prefix + '.query.h5'
        if os.path.isfile(query_fname):
            os.remove(query_fname)
        videoextractor.write_database(query_fname, args.input, frame_step=args.frame_step, start_frame=args.start_frame, end_frame=args.end_frame)
        query_features = h5autoclose(h5py.File(query_fname, 'r'))
    else:
        query_features = videoextractor.inmem_database(args.input, frame_step=args.frame_step, start_frame=args.start_frame, end_frame=args.end_frame)
    info = read_video_params(args.input)
    resolution = [info['width'], info['height']]

logger.info("Matching stage...")
matcher = SuperGluePrefilteringMatcher(matcher_config=matcher_config)

torch.cuda.empty_cache()

query_pairs = matcher.prefilter(db_features, query_features)
if args.filecache_prefix:
    matches_fname = args.filecache_prefix + '.matches.h5'
    if os.path.isfile(matches_fname):
        os.remove(matches_fname)
    matcher.write_matches(matches_fname, db_features, query_features, query_pairs)
    query_matches = h5autoclose(h5py.File(matches_fname, 'r'))
else:
    query_matches = matcher.inmem_match(db_features, query_features, query_pairs)

logger.info("Localization stage...")
camera_model = {
    'model': colmap_models[args.camera_model],
    'width': resolution[0],
    'height': resolution[1],
    'params': camera_params
}

localizer = Localizer(camera_model, min_matches=20, ransac_max_error=args.ransac_max_error)
results = localizer.localize(query_matches, query_features, db_features)
logger.info(f"Done. Writing results to {args.outfile}")
s_results = {os.path.splitext(k)[0]: {kk: vv.tolist() for kk, vv in v.items()} if v is not None else None for k, v in results.items()}
os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
json.dump(s_results, open(args.outfile, 'w'))
