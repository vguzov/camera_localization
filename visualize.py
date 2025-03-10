from argparse import ArgumentParser
from loguru import logger
from ocam_renderer import PointCloudRenderer
from ocam_renderer.libegl import EGLContext
from ocam_renderer.utils import load_pc_from_zip
import json
import numpy as np
from tqdm import tqdm
from videoio import VideoWriter, VideoReader, read_video_params
from queue import SimpleQueue

# Add known cameras here for convenience
known_cameras = {
    'example_camera1': {'camera_model': 'opencv',
                        'camera_params': [871.80021916157159, 885.57997989678529, 961.54424567006754, 550.68646879951450,
                                          -0.25462584991295, 0.08039095012756, 0.00014583290360, -0.00001397345667]},
    'example_camera2': {'camera_model': 'opencv',
                        'camera_params': [870.39802320390481, 883.77772671373339, 977.24123059216777, 550.42154564060434,
                                          -0.26512391289441, 0.09675221734815, 0.00000604206872, 0.00012467424427]},
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-iv", "--input_video", required=True, help="Path to input video")
    parser.add_argument("-il", "--input_loc", required=True, help="Path to localization results")
    parser.add_argument("-iz", "--input_pczip", required=True, help="Path to zipped scene scan")
    parser.add_argument("-o", "--output_video", required=True, help="Path to output video")
    parser.add_argument("-cm", "--camera_model", default="opencv", choices=['opencv', 'ocam'] + list(known_cameras.keys()))
    parser.add_argument("-cp", "--camera_params", nargs='*', type=float, help="Camera parameters in OpenCV format: fx fy cx cy k1 k2 p1 p2. "
                                                                              "If camera_model is from known_cameras, this argument is ignored.")
    parser.add_argument('-fc', '--frames_cap', default=None, type=int, help="Maximum number of frames to render")
    parser.add_argument('--far', type=float, default=100., help="Maximum rendering distance (meters)")
    parser.add_argument("-oi", "--overlay_input", choices=["right", "left", "none"], default="left", help="Overlay input image")

    args = parser.parse_args()
    logger.info(f"Loading pointcloud")

    pointcloud = load_pc_from_zip(args.input_pczip, "pointcloud.ply")

    videoparams = read_video_params(args.input_video)
    resolution = (videoparams['width'], videoparams['height'])
    logger.info(f"Rendering at {resolution} resolution")

    ctx = EGLContext()

    if not ctx.initialize(*resolution):
        logger.error('Could not initialize OpenGL context.')

    opencv_renderer = PointCloudRenderer(*resolution)
    opencv_renderer.init_opengl()

    if args.camera_model in known_cameras:
        camera = known_cameras[args.camera_model]
    else:
        camera = {"camera_model": args.camera_model,
                  "camera_params": args.camera_params}

    focal_dist = camera['camera_params'][:2]
    center = camera['camera_params'][2:4]
    dist_coeffs = camera['camera_params'][4:] + [0.]

    opencv_renderer.init_context(pointcloud, camera['camera_model'], image_size=resolution, focal_dist=focal_dist,
                                 center=center,
                                 distorsion_coeffs=dist_coeffs, far=args.far)

    tqdm_iter = tqdm(VideoReader(args.input_video)) if not args.frames_cap else tqdm(VideoReader(args.input_video), total=args.frames_cap)
    s_results = json.load(open(args.input_loc))

    last_frame = None
    frame_queue = SimpleQueue()
    queue_size = 50
    with VideoWriter(args.output_video, resolution=resolution, fps=30, preset='veryfast') as vw:
        def process_frame(delete_pbo=True):
            if frame_queue.qsize() >= queue_size:
                last_frame = frame_queue.get()
            else:
                last_frame = None
            if last_frame is not None:
                prev_orig_color, pbo, active = last_frame
                if not active:
                    color = np.zeros(resolution[::-1] + (3,), dtype=np.uint8)
                else:
                    color = opencv_renderer.get_requested_color(pbo, delete_pbo=delete_pbo)
                    color = color[::-1]
                if args.overlay_input != "none":
                    if args.overlay_input == "right":
                        color = np.hstack([color[:, :resolution[0] // 2],
                                           prev_orig_color[:, resolution[0] // 2:]])
                    else:
                        color = np.hstack([prev_orig_color[:, :resolution[0] // 2],
                                           color[:, resolution[0] // 2:]])
                vw.write(color)
                return pbo
            else:
                return None


        for frame_ind, orig_color in enumerate(tqdm_iter):
            pbo = process_frame(False)
            if args.frames_cap and frame_ind >= args.frames_cap:
                break
            imname = f"{frame_ind:06d}"
            if imname not in s_results: # Check for old naming convention
                imname = str(frame_ind)
            impos = s_results[imname] if imname in s_results else None
            if impos is not None:
                pos = np.array(impos['position'])
                quat = np.array(impos['quaternion'])
                opencv_renderer.locate_camera(quat, pos)
                opencv_renderer.draw()
                pbo = opencv_renderer.request_color_async(pbo)
                active = True
            else:
                active = False
            frame_queue.put((orig_color, pbo, active))
        queue_size = 0
        while not frame_queue.empty():
            process_frame(True)
