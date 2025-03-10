import os
from argparse import ArgumentParser

from hloc_standalone.extractor import SuperPointNetVladExtractor

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
            'batch_size': 1
        },
        'storage': {
            'as_half': False,
        }
    },
    'global': {
        'model': {},
        'loader': {
            'workers': 8,
            'batch_size': 1
        }
    }
}

parser = ArgumentParser()
parser.add_argument("-i", "--input_renders_dir", required=True, help="Path to scan renders directory")
parser.add_argument("-o", "--output", required=True, help="Path to output database")
args = parser.parse_args()

extractor = SuperPointNetVladExtractor(extractor_config=extractor_config)

extractor.write_database(args.output, os.path.join(args.input_renders_dir, 'undistorted_orig'),
                         vertex_mapping_dir=os.path.join(args.input_renders_dir, 'undistorted'))
