import os
import numpy as np
from pycolmap import estimate_and_refine_absolute_pose, AbsolutePoseEstimationOptions, AbsolutePoseRefinementOptions, Camera
from tqdm import tqdm
from .tools import interpolate_scan
from scipy.spatial.transform import Rotation
from loguru import logger


class Localizer:
    def __init__(self, camera_model, min_matches = 20, ransac_max_error = 48.00, show_progress = True):
        """
        Args:
            camera_model (dict): camera model description
            min_matches: minimal matches count for candidate image to be added to the set
            ransac_max_error: error for absolute pose estimation in pixels
        """
        self.camera_model = Camera(camera_model)
        self.min_matches = min_matches
        self.pose_estimation_options = AbsolutePoseEstimationOptions()
        self.pose_estimation_options.ransac.max_error = ransac_max_error
        self.pose_refinement_options = AbsolutePoseRefinementOptions()
        self.show_progress = show_progress

    @staticmethod
    def invert_extrinsics(pos_dict):
        rot = Rotation.from_quat(np.roll(pos_dict['quaternion'], -1))
        invrot = rot.inv()
        invpos = -invrot.as_matrix().dot(pos_dict['position'])
        return ({'quaternion': np.roll(invrot.as_quat(), 1), 'position': invpos})

    @property
    def ransac_max_error(self):
        return self.pose_estimation_options.ransac.max_error

    def localize(self, match_db, query_feats, database_feats, vertex_mapping_dir = None, invert = True):
        results = {}
        progress_func = tqdm if self.show_progress else (lambda x: x)
        for query_name in progress_func(match_db.keys()):
            matched_from_database = match_db[query_name]
            curr_query_keypoints = query_feats[query_name]['keypoints'].__array__()
            all_query_kp = []
            all_db_kp3d = []
            all_indices = []
            for database_ind, database_name in enumerate(matched_from_database.keys()):
                current_match = matched_from_database[database_name]
                matches = current_match['matches'].__array__()
                valid = (matches > -1)
                if np.count_nonzero(valid) < self.min_matches:
                    continue
                valid_curr_query_keypoints = curr_query_keypoints[valid]

                if vertex_mapping_dir is None and 'keypoints3d' in database_feats[database_name]:
                    valid_curr_db_keypoints3d = database_feats[database_name]['keypoints3d'].__array__()[matches[valid]]
                    valid3d = np.isfinite(valid_curr_db_keypoints3d).all(axis=1)
                else:
                    assert vertex_mapping_dir is not None
                    curr_db_keypoints = database_feats[database_name]['keypoints'].__array__()
                    valid_curr_db_keypoints = curr_db_keypoints[matches[valid]]
                    scan_r = np.load(os.path.join(vertex_mapping_dir,
                                                  os.path.splitext(database_name)[0] + '.xyz.npz'))['xyz']
                    valid_curr_db_keypoints3d, valid3d = interpolate_scan(scan_r, valid_curr_db_keypoints)
                all_query_kp.append(valid_curr_query_keypoints[valid3d])
                all_db_kp3d.append(valid_curr_db_keypoints3d[valid3d])
                all_indices.append(np.full(np.count_nonzero(valid3d), database_ind))
            if len(all_query_kp) > 0:
                all_query_kp = np.concatenate(all_query_kp, 0)
                all_db_kp3d = np.concatenate(all_db_kp3d, 0)
                all_indices = np.concatenate(all_indices, 0)
                all_db_kp3d = all_db_kp3d.astype(np.float32).copy()
                all_query_kp = all_query_kp.astype(np.float32).copy()
                ret = estimate_and_refine_absolute_pose(
                    all_query_kp, all_db_kp3d, self.camera_model, self.pose_estimation_options, self.pose_refinement_options)
                if ret is not None:
                    res = {'quaternion':np.roll(ret['cam_from_world'].rotation.quat, 1), 'position':ret['cam_from_world'].translation}
                    if invert:
                        res = self.invert_extrinsics(res)
                else:
                    logger.warning("{} is not localized (ransac error)".format(query_name))
                    res = None
                results[query_name] = res
            else:
                logger.warning("{} is not localized".format(query_name))
                results[query_name] = None
        return results




