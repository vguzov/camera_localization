import os
import numpy as np
import json
import xmltodict
import cv2
from copy import deepcopy
from tqdm import tqdm
from argparse import ArgumentParser
from multiprocessing import Process, Queue
from loguru import logger

from ocam_renderer import PointCloudRenderer
from ocam_renderer.ocamcamera import OcamCamera
from ocam_renderer.utils import load_pc_from_zip, open_from_zip, zip_glob
from ocam_renderer.libegl import EGLContext



def compute_undist_mapping(cmodel, undist_resolution=(2000, 2000), fov=360):
    w, h = undist_resolution
    cmodel = deepcopy(cmodel)
    cmodel['OCamModel']['cx'], cmodel['OCamModel']['cy'] = cmodel['OCamModel']['cy'], cmodel['OCamModel']['cx']
    ocamera = OcamCamera(cmodel, fov=fov)
    z = w / 3.0
    x = [i - w / 2 for i in range(w)]
    y = [j - h / 2 for j in range(h)]
    x_grid, y_grid = np.meshgrid(x, y, sparse=False, indexing='xy')
    point3D = np.stack([x_grid, y_grid, np.full_like(x_grid, z)]).reshape(3, -1)
    mapx, mapy = ocamera.world2cam(point3D)
    mapx = mapx.reshape(h, w)
    mapy = mapy.reshape(h, w)
    return mapx, mapy


def saving_process():
    while True:
        item = mp_queue.get()
        if item is None:
            return
        img, mapping, frame_id, cameraname, undist_map = item
        img = img.transpose(1, 0, 2).copy()
        mapping = mapping.T.copy()
        fname_dist = os.path.join(outdir, "distorted", f"{frame_id:05d}-{cameraname}")
        cv2.imwrite(fname_dist + '.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        np.savez_compressed(fname_dist + '.npz', map=mapping)
        fname_undist = os.path.join(outdir, "undistorted", f"{frame_id:05d}-{cameraname}")
        undist_mapx, undist_mapy = undist_map
        img = cv2.remap(img, undist_mapx, undist_mapy, cv2.INTER_LINEAR)
        mapping = cv2.remap(mapping, undist_mapx, undist_mapy, cv2.INTER_NEAREST)
        cv2.imwrite(fname_undist + '.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        np.savez_compressed(fname_undist + '.npz', map=mapping)
        orig = cv2.imdecode(np.frombuffer(open_from_zip(args.inzip, f"cam/{frame_id:05d}-{cameraname}.jpg").read()), cv2.IMREAD_COLOR)
        orig = cv2.remap(orig, undist_mapx, undist_mapy, cv2.INTER_LINEAR)
        fname_undist_orig = os.path.join(outdir, "undistorted_orig", f"{frame_id:05d}-{cameraname}")
        cv2.imwrite(fname_undist_orig + '.jpg', orig)

        # XYZ generation
        mp = mapping
        mshape = mp.shape
        mp = mp.flatten()
        mp_invalid = mp < 0
        pointsmap = pointcloud.vertices[mp.flatten()]
        pointsmap[mp_invalid] = float('nan')
        pointsmap = pointsmap.reshape(mshape + (3,))
        outpath = fname_undist + ".xyz.npz"
        np.savez_compressed(outpath, xyz=pointsmap)


parser = ArgumentParser()
parser.add_argument("-i", "--inzip", required=True, help="Path to the HPS/iReplica full scan zip file")
parser.add_argument("-o", "--outdir", required=True, help="Path to folder to save the output renders to")
parser.add_argument("-t", "--threads", type=int, default=None, help="How many parallel processes to use")
parser.add_argument("-ures", "--undist_resolution", type=int, nargs=2, default=(2000, 2000), help="Resolution of the undistorted image")
parser.add_argument("-cam", "--camera_views", type=int, default=6, help="How many camera views per scanner frame are in the archive")

args = parser.parse_args()

outdir = args.outdir

os.makedirs(os.path.join(outdir, "undistorted"), exist_ok=True)
os.makedirs(os.path.join(outdir, "undistorted_orig"), exist_ok=True)
os.makedirs(os.path.join(outdir, "distorted"), exist_ok=True)

sensor_frame = xmltodict.parse(open_from_zip(args.inzip, "sensor_frame.xml"))
cmodel = sensor_frame['SensorFrame']['CameraHead']['CameraModel'][0]
ocam_imsize = cmodel['ImageSize']
ocam_img_size = np.array((int(ocam_imsize['Width']), int(ocam_imsize['Height'])))

resolution = ocam_img_size

logger.info("Loading pointcloud... ")
pointcloud = load_pc_from_zip(args.inzip, "pointcloud.ply")
pc_max = pointcloud.vertices.max(axis=0)
pc_min = pointcloud.vertices.min(axis=0)
pc_len = np.linalg.norm(pc_max - pc_min)
far = pc_len + 0.5
logger.info(f"Pointcloud size is {(pc_max - pc_min).tolist()}, far is {far}")

logger.info("Setting up the renderer")
with EGLContext() as ctx:
    if not ctx.initialize(*resolution):
        logger.error('Could not initialize OpenGL context.')

    ocam_renderer = PointCloudRenderer(*resolution)
    ocam_renderer.init_opengl()

    frames = {}
    try:
        cam_info = open_from_zip(args.inzip, "cam/info.list").read().decode("utf-8").split("\n")
    except FileNotFoundError: # if not info file exists, traverse the archive
        frame_inds = sorted(set([int(os.path.basename(x).split("-cam")[0]) for x in zip_glob(args.inzip, "cam/*.jpg")]))
        for frame_ind in frame_inds:
            frames[frame_ind] = json.load(open_from_zip(args.inzip, f"info/{frame_ind:05d}-info.json"))
    else:
        for line in cam_info:
            line = line.strip()
            if len(line) > 0:
                frame_ind = int(line.split('-')[0])
                frames[frame_ind] = json.load(open_from_zip(args.inzip, f"info/{frame_ind:05d}-info.json"))
    logger.info(f"Read {len(frames)} frames")

    cmodel = [c for c in sensor_frame['SensorFrame']['CameraHead']['CameraModel'] if c['SensorName'] == "cam0"][0]
    ocam_renderer.init_context(pointcloud, "ocam", cameramodel_dict=cmodel)

    saving_threads = []
    threads = os.cpu_count() if args.threads is None else args.threads
    mp_queue = Queue(maxsize=int(1.5 * threads))
    for i in range(threads):
        p = Process(target=saving_process)
        p.start()
        saving_threads.append(p)

    dist = 30.0
    fov = 360
    camera_views = args.camera_views

    logger.info("Rendering frames")
    with tqdm(total=camera_views * len(frames)) as tqdm_iter:
        for cam_ind in range(camera_views):
            camera = f'cam{cam_ind}'
            cmodel = [c for c in sensor_frame['SensorFrame']['CameraHead']['CameraModel'] if c['SensorName'] == camera][0]
            ocam_renderer.camera.init_intrinsics(cameramodel_dict=cmodel, far=far, fov=fov)
            undist_map = compute_undist_mapping(cmodel, args.undist_resolution, fov=fov)

            for frame_ind, frame in sorted(frames.items()):
                ocam_renderer.locate_camera(quat=frame[camera]['quaternion'], pose=frame[camera]['position'])
                ocam_renderer.draw()
                color, mapping = ocam_renderer.get_image()
                mp_queue.put((color, mapping, frame_ind, camera, undist_map))
                tqdm_iter.update(1)

    for t in saving_threads:
        mp_queue.put(None)
    logger.info("Waiting for saving processes to finish")
    for t in saving_threads:
        t.join()
