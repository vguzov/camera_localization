import torch
import h5py
import os
import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy
from loguru import logger
from .third_party.SuperGluePretrainedNetwork.models.superglue import SuperGlue as SG
from .tools import h5autoclose


class SuperGluePrefilteringMatcher:
    default_config = {
        'matcher': {
            'model': {
                'name': 'superglue',
                'weights': 'outdoor',
                'sinkhorn_iterations': 50,
            }
        },
        'filter': {
            'matches_per_query': 30
        }
    }

    def __init__(self, device=None, matcher_config=None, show_progress=True):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        config = deepcopy(self.default_config)
        if matcher_config is not None:
            config.update(matcher_config)
        self.config = config
        self.matcher_model = SG(config['matcher']['model']).eval().to(device)
        self.show_progress = show_progress

    def prefilter(self, database_feats, query_feats):
        def tensor_from_names(feats, names):
            desc = [feats[i]['global_descriptor'].__array__() for i in names]
            desc = torch.from_numpy(np.stack(desc, 0)).to(self.device).float()
            return desc

        database_names = list(database_feats.keys())
        query_names = list(query_feats.keys())
        database_desc = tensor_from_names(database_feats, database_names)
        query_desc = tensor_from_names(query_feats, query_names)
        matches_per_query = self.config['filter']['matches_per_query']
        sim = torch.einsum('id,jd->ij', query_desc, database_desc)
        topk = torch.topk(sim, matches_per_query, dim=1).indices.cpu().numpy()
        query_pairs = {}
        for query_name, indices in zip(query_names, topk):
            query_pairs[query_name] = []
            for i in indices:
                query_pairs[query_name].append(database_names[i])
        return query_pairs

    def match_points(self, output_db, database_feats, query_feats, query_pairs = None):
        if query_pairs is None:
            logger.info("No pairs was provided, starting exhaustive matching...")
            database_names = list(database_feats.keys())
            query_names = list(query_feats.keys())
            query_pairs = self._generate_exhaustive_pairs(database_names, query_names)
        data = {}
        progress_func = tqdm if self.show_progress else (lambda x: x)
        with torch.no_grad():
            for query_name, db_names in progress_func(query_pairs.items()):
                curr_query_feats = query_feats[query_name]
                for k in curr_query_feats.keys():
                    v = curr_query_feats[k].__array__()
                    v = torch.from_numpy(v)[None].float().to(self.device)
                    data[k + '0'] = v
                data['image0'] = torch.empty((1, 1,) + tuple(curr_query_feats['image_size'])[::-1])
                query_grp = output_db.create_group(query_name)
                for db_name in progress_func(db_names):
                    curr_db_feats = database_feats[db_name]
                    for k in curr_db_feats.keys():
                        v = curr_db_feats[k].__array__()
                        v = torch.from_numpy(v)[None].float().to(self.device)
                        data[k + '1'] = v
                    data['image1'] = torch.empty((1, 1,) + tuple(curr_db_feats['image_size'])[::-1])
                    pred = self.matcher_model(data)
                    pair_grp = query_grp.create_group(db_name)
                    matches = pred['matches0'][0].cpu().short().numpy()
                    pair_grp.create_dataset('matches', data=matches)
                    if 'matching_scores0' in pred:
                        scores = pred['matching_scores0'][0].cpu().half().numpy()
                        pair_grp.create_dataset('matching_scores', data=scores)
            logger.info('Finished exporting matches.')
        return output_db

    def write_matches(self, matches_path, database_feats, query_feats, query_pairs = None):
        os.makedirs(os.path.dirname(matches_path), exist_ok=True)
        desc_file = h5py.File(str(matches_path), 'w')
        self.match_points(desc_file, database_feats, query_feats, query_pairs)
        desc_file.close()

    def _generate_exhaustive_pairs(self, db_keys, query_keys):
        pairs = {}
        db_keys_list = list(db_keys)
        for qk in query_keys:
            pairs[qk] = db_keys_list
        return pairs

    def inmem_match(self, database_feats, query_feats, query_pairs = None, cache_path = None):
        if cache_path is None:
            desc_file = h5py.File('tmp_matches.{}.h5'.format(random.randint(0,65536)), 'w', driver='core', backing_store=False)
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            desc_file = h5py.File(cache_path, 'w', driver='core')
        desc_file = h5autoclose(desc_file)
        self.match_points(desc_file, database_feats, query_feats, query_pairs)
        return desc_file


