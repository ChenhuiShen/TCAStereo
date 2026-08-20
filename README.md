# TCAStereo

Complete tri-camera data flow:

* The third RGB/texture camera is geometrically related to the left infrared camera through an external rigid transformation (RT).
* GEM enhances geometric structures at four scales—1/4, 1/8, 1/16, and 1/32—using Gaussian high-pass filtering, FFT, and four learnable complex-valued frequency-domain kernels.
* GAM converts the initial stereo disparity into metric depth, projects it into the texture camera, and uses RANSAC to estimate the global scale `alpha` and bias `beta` of the relative depth predicted by Depth Anything V2. The aligned depth is then reprojected into the left-camera coordinate system.
* During each recurrent update, FIU generates spatial attention from the aligned monocular depth and uses it to gate the original stereo residual disparity.
* Depth Anything V2 uses the official implementation with frozen weights and supports either online inference or offline monocular-depth caching.

## Project Structure

```text
core/TCAStereo.py                 Main tri-camera network
core/tca/gem.py                   Geometric Enhancement Module
core/tca/gam.py                   Global Alignment Module, RANSAC, and projection
core/tca/fiu.py                   Fusion Iteration Unit
core/tca/calibration.py           Intrinsics, external RT, units, and direction handling
core/tca/depth_anything.py        Frozen Depth Anything V2 adapter
core/tri_camera_dataset.py        JSONL-based tri-camera training dataset
infer_tca.py                      Tri-camera inference
train_tca.py                      Tri-camera training
configs/                          Calibration, RT, and manifest examples
third_party/Depth-Anything-V2/    Official Depth Anything V2 source code
tests/test_tca_modules.py         Geometry, gradient, and interface unit tests
```

## 1. Installation

It is recommended to create a CUDA-enabled PyTorch environment before installing the dependencies:

```powershell
pip install -r requirements.txt
```

The official Depth Anything V2 source code is included in this project, but the pretrained model weights are not provided. Download the ViT-L checkpoint used in the paper.

The ViT-S weights are licensed under Apache-2.0, whereas the ViT-B and ViT-L weights are licensed under CC-BY-NC-4.0. Before industrial or commercial deployment, verify the applicable terms in the official licenses.

## 2. Calibration and External RT

The following transformation convention is used internally:

```text
X_texture = T_left_to_texture * X_left
```

Copy and modify the following example files:

* `configs/calibration.example.json`: intrinsics of the left and texture cameras, stereo baseline, and calibration image resolution;
* `configs/rt_left_to_texture.example.json`: external RT for the third camera.

If your transformation matrix maps the texture camera to the left camera, specify the following in the RT file:

```json
{"direction": "texture_to_left"}
```

The loader will automatically invert the transformation. The RT loader supports JSON, YAML, NPY, NPZ, and 3×4 or 4×4 TXT files. JSON and YAML files may specify `translation_unit` as `m`, `cm`, or `mm`. All values are converted to meters internally.

`disparity_offset` denotes the full-resolution disparity offset corresponding to the plane at infinity. For a negative-disparity convention, set `disparity_sign: -1`. Both parameters must be consistent with the current stereo rectification and disparity definition.

## 3. Inference

```powershell
python infer_tca.py `
  --left "data/test/left/*.png" `
  --right "data/test/right/*.png" `
  --texture "data/test/rgb/*.png" `
  --calibration configs/calibration.json `
  --rt configs/rt_left_to_texture.json `
  --checkpoint checkpoints/tca/tca_step_0080000.pth `
  --depth-anything-checkpoint checkpoints/depth_anything_v2/depth_anything_v2_vitl.pth `
  --mixed-precision `
  --output-directory out/tca
```

By default, inference performs 32 FIU iterations, consistent with the setting used in the paper. The following files are generated for each sample:

* `*_disparity.npy/.pfm/.png`: final disparity;
* `*_aligned_depth.npy/.png`: monocular depth aligned by GAM and transformed into the left-camera coordinate system;
* `*_alignment.json`: `alpha`, `beta`, projection coverage, and runtime statistics.

If the monocular prior has already been generated offline, use `--mono-depth "cache/*.npy"`. In this case, the Depth Anything V2 weights do not need to be loaded.

## 4. Training

The training manifest uses the JSONL format, with one synchronized sample per line. See `configs/train_manifest.example.jsonl`:

```json
{"left":"left.png","right":"right.png","texture":"rgb.png","disparity":"disp.pfm","calibration":"calibration.json","rt":"rt.json"}
```

The `calibration` and `rt` files can be specified individually for each sample or globally through command-line arguments. Because the texture camera does not share the same pixel coordinate system as the left camera, training crops are applied only to the left and right infrared images and the ground-truth disparity. The principal point of the left camera is updated accordingly. The complete field of view of the texture image is retained, and GAM performs fusion through explicit geometric projection.

```powershell
python train_tca.py `
  --manifest datasets/speck3d/train.jsonl `
  --depth-anything-checkpoint checkpoints/depth_anything_v2/depth_anything_v2_vitl.pth `
  --restore-checkpoint checkpoints/stereo_pretrain.pth `
  --image-size 256 512 `
  --batch-size 4 `
  --train-iters 22 `
  --num-steps 20000 `
  --mixed-precision
```

The training script uses AdamW with `beta=(0.9, 0.999)` and a weight decay of `1e-4`, together with polynomial learning-rate decay, initial smooth-L1 supervision, and iterative L1 supervision. Depth Anything V2 remains frozen throughout training.

A legacy SpeckleStereo checkpoint can be loaded for initialization, allowing the stereo backbone parameters to be reused. GEM and FIU contain newly introduced parameters and must therefore be trained or fine-tuned using tri-camera data. If the inference log reports missing parameters for these modules, results produced with randomly initialized parameters must not be treated as representative of the performance reported in the paper.

## 5. Testing

```powershell
python -m py_compile core/TCAStereo.py core/tca/*.py train_tca.py infer_tca.py
```

The unit tests cover RT direction and unit handling, RANSAC-based global alignment, GEM gradient propagation, FIU valid-region handling, and the Depth Anything V2 tensor interface.

## 6. Practical Data Acquisition Considerations

* The three cameras must be hardware-synchronized, or their temporal offset must be negligible. GAM cannot compensate for temporal misalignment in dynamic scenes.
* The RT should be obtained through joint calibration between the left infrared camera and the texture camera. The intrinsics must correspond to the currently rectified left camera.
* The stereo `baseline`, RT translation, and ground-truth depth must use consistent units. The example configuration accepts translations in millimeters, which are converted to meters at runtime.
* GAM relies on the initial stereo depth to establish correspondences for RANSAC. When severe reflections, extensive occlusions, or large calibration errors are present, inspect the projection coverage and `alpha/beta` values recorded in the output JSON file.
* The performance reported in the paper depends on training with SPECK3D/SceneFlow and using the corresponding paper checkpoint. Implementing only the network structure does not automatically reproduce the reported accuracy.
