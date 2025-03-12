# RGB camera self-localization framework
This framework was used in [HPS](https://virtualhumans.mpi-inf.mpg.de/hps/) and [iReplica](https://virtualhumans.mpi-inf.mpg.de/ireplica/) projects 
for initial head camera localization. This framework is based on [hloc - the hierarchical localization toolbox](https://github.com/cvg/Hierarchical-Localization) repository 
and uses code from [SuperGlue]( https://github.com/magicleap/SuperGluePretrainedNetwork.git) and [NetVLAD](https://github.com/Nanne/pytorch-NetVlad) 
PyTorch implementations. Tested on Ubuntu 24.04.

## Installation
Clone the repository, make sure to clone the submodules as well:
```bash
git clone --recursive https://github.com/vguzov/cameraloc.git
```

Install the dependencies:
```bash
pip install -r requirements.txt
```
or, if you want to reproduce the environment with exact package versions:
```bash
pip install -r requirements_strict.txt
```

Get the pretrained NetVLAD model weights:
Go to https://github.com/Nanne/pytorch-NetVlad and download the checkpoint from "Additionally, the model state for the above run is available here" link.
Extract the file `checkpoint.pth.tar` from `vgg16_netvlad_checkpoint/checkpoints` and place it in the `hloc_standalone/netvlad/weights` folder.


## Usage
### Generate the feature database for anchor images
Both HPS and iReplica scenes have 3D pointclouds and RGB images. If you have a similar dataset, you can use this framework to generate the feature database for anchor images.

<details>
<summary>Dataset structure</summary>
Each scene for the dataset is a zip archive which contains (at least) these files:

- pointcloud.ply - Scene 3D scan in a form of a point cloud, evenly subsampled to 5mm distance between the points
- pointcloud-raw.ply - Denser version of the same scan
- cam/ - Folder with original camera images captured during the scanning process
  - 00000-cam0.jpg - frame 0, camera 0
  - 00000-cam1.jpg - frame 0, camera 1
  - ...
- info/ - Postion of each camera, per frame
  - 00000-info.json - dict with the following fields:
    - "cam0": {"position": 3D translation, "rotation": WXYZ quaternion}
    - ...
    - "camN": ...
- "sensor_frame.xml" - XML file with calibration info for each camera. The camera calibration follows [OCamCalib model](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab). The minimal file structure is:
  - SensorFrame
    - CameraHead - contains several "CameraModel" tags, one for each camera of the scanner
      - CameraModel
        - SensorName ("cam0" .. "camN")
        - OCamModel
          - cam2world
            - coeff - list of polynomial coefficients for the direct mapping function
          - world2cam
            - coeff - list of polynomial coefficients for the inverse mapping function
          - cx - camera center X
          - cy - camera center Y
          - c - affine parameter C
          - d - affine parameter D
          - e - affine parameter E
        - ImageSize
          - Width
          - Height
</details>

This step is aimed to extract the key points and their descriptors from RGB images and 
determine the 2D-3D correspondences between each key point and the 3D pointcloud.

First, we need to render the 3D pointclouds from the same viewpoint and view the same intrinsics as the RGB images. Additionally, we need to undistort the RGB images to improve the feature extraction quality.
This can be done using the `ocam_render_and_undistort.py` script. For each scan, run:

**Note:** On Wayland systems (e.g. Ubuntu 24+), troubles with offscreen EGL rendering might occur, e.g. `OpenGL.error.Error: Attempt to retrieve context when no valid context`. 
Try running the script with `PYOPENGL_PLATFORM=egl` prepended, e.g. `PYOPENGL_PLATFORM=egl python ocam_render_and_undistort.py ...`.

```bash
python ocam_render_and_undistort.py --pc_zip <path to the zipped 3D pointcloud> --info_zip <zip file containing frames and calibration information> --info_folder <path inside info_zip containing calibration information> --outdir <output folder> [-t <number of threads>] [-ures <undistorted frames resolution>]
```

This script renders the 3D pointclouds from the same viewpoint, 
undisort the original and rendered frames and save them to the output folder together with the per-pixel 3D correspondences.
As a result, `outdir` contains 3 folders, `distorted` (rendered result), `undistorted` (undistorted result) and `undistorted_orig` (undistorted original frames).
`undistorted` folder additionally contains the `<name>.xyz.npz` files with the 3D correspondences and `<name>.npz` files with pixel-to-point-index mapping.

<details>
<summary>Real vs rendered image comparison</summary>

<img alt="real_vs_rendered" src=.github/images/real_vs_rendered.gif width="400">

</details>


Then, we can extract the features and descriptors from the undistorted RGB images and save them to the database.

```bash
python create_feature_db.py -i <path to the output folder from the previous step> -o <output database file .h5>
```

The resulting database contains the key points, their descriptors, and the 2D-3D correspondences.

### Localization
To localize the camera in the scene, you can use the `localize.py` script to extract the features and descriptors from the input images (or video) and match them with the anchor images from the database.
```bash
python localize.py -i <folder to images or path to a video> -db <database file from the previous step> -o <output localization file> -cm <camera model (opencv, perspective, opencv_fisheye)> -cp <camera params, space separated>
```
The script has additional optional parameters, which can be found by running `python localize.py -h`.

The script outputs the localization results in the form of a JSON file, containing the camera poses and the corresponding frame names (frame numbers in case of the video).

Note that localization results might be quite noisy, especially for the head-mounted cameras; 
consider applying velocity-based filtering and interpolation; implementation examples can be found [here](hloc_standalone/filtering.py).

**Tip for parallel execution:** When localizing a video, you can run several instances of `localize.py` independently on different machines. 
For that, run each localizer with larger `frame_step` to localize every n-th frame per process and supply different `--start_frame` for each. 
After that, you can concatenate resulting JSONs together and get the complete localization result.

### Visualization
If you used video as an input, you can run the `visualize.py` script to render out and visualize the result.
```bash
python visualize.py -iv <path to input video> -il <resulting localization> -iz <input zipped scene scan> -o <output video> -cm <camera model (opencv, ocam)> -cp <camera params, space separated>
```
**Note:** On Wayland systems (e.g. Ubuntu 24+), troubles with offscreen EGL rendering might occur, e.g. `OpenGL.error.Error: Attempt to retrieve context when no valid context`. 
Try running the script with `PYOPENGL_PLATFORM=egl` prepended, e.g. `PYOPENGL_PLATFORM=egl python visualize.py ...`.

<details>
<summary>Result example (real video on the left, rendered on the right)</summary>

<img alt="visualization_example" src=.github/images/visualization_example.gif width="480">

</details>

## Data
The scans used to localize the camera in the HPS and iReplica projects are released as a part of the corresponding datasets and can be found [here](https://virtualhumans.mpi-inf.mpg.de/hps/hps_ireplica_scenes.html).

## Citation

If you find this code useful, please consider citing our papers:

```bibtex
@inproceedings{guzov-mir2021human,
  title={Human POSEitioning System (HPS): 3D Human Pose Estimation and Self-localization in Large Scenes from Body-Mounted Sensors},
  author={Guzov, Vladimir and Mir, Aymen and Sattler, Torsten and Pons-Moll, Gerard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4318--4329},
  year={2021}
}
```

```bibtex
@inproceedings{guzov24ireplica,
    title = {Interaction Replica: Tracking human–object interaction and scene changes from human motion},
    author = {Guzov, Vladimir and Chibane, Julian and Marin, Riccardo and He, Yannan and Saracoglu, Yunus and Sattler, Torsten and Pons-Moll, Gerard},
    booktitle = {International Conference on 3D Vision (3DV)},
    month = {March},
    year = {2024},
}
```