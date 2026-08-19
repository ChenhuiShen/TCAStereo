# TCAStereo：三相机主动双目 + 单目先验融合

完整三相机数据流：

- 第三台 RGB/纹理相机通过外部 RT 与左红外相机建立坐标变换；
- GEM 在 1/4、1/8、1/16、1/32 四尺度用高斯高通、FFT 和 4 个可学习复数频域核增强几何结构；
- GAM 将初始双目视差转为度量深度，投影到纹理相机，用 RANSAC 求解 Depth Anything V2 相对深度的全局尺度 `alpha` 与偏置 `beta`，再反投影到左相机；
- FIU 在每次循环更新中，用对齐后的单目深度生成空间注意力，对原 stereo residual disparity 进行门控；
- Depth Anything V2 使用官方源码和冻结权重，支持在线推理或离线缓存单目深度。


## 目录

```text
core/TCAStereo.py                 三相机主网络
core/tca/gem.py                   Geometric Enhancement Module
core/tca/gam.py                   Global Alignment Module + RANSAC + 投影
core/tca/fiu.py                   Fusion Iteration Unit
core/tca/calibration.py           内参、外部 RT、单位和方向处理
core/tca/depth_anything.py        Depth Anything V2 冻结适配器
core/tri_camera_dataset.py        JSONL 三相机训练数据集
infer_tca.py                      三相机推理
train_tca.py                      三相机训练
configs/                          标定、RT 和 manifest 示例
third_party/Depth-Anything-V2/    官方 Depth Anything V2 源码
tests/test_tca_modules.py         几何、梯度和接口单测
```

## 1. 安装

建议创建带 CUDA 的 PyTorch 环境，再安装依赖：

```powershell
pip install -r requirements.txt
```

工程已经包含官方 Depth Anything V2 源码，但不包含大模型权重。下载论文使用的 ViT-L：

ViT-S 权重采用 Apache-2.0；ViT-B/ViT-L 权重采用 CC-BY-NC-4.0。工业商业部署前请按官方许可重新确认使用范围。

## 2. 标定与外部 RT

内部统一采用：

```text
X_texture = T_left_to_texture * X_left
```

复制并修改：

- `configs/calibration.example.json`：左相机内参、纹理相机内参、双目 baseline、图像标定分辨率；
- `configs/rt_left_to_texture.example.json`：第三相机外部 RT。

如果手里的矩阵是 `texture_to_left`，可在 RT 文件中写：

```json
{"direction": "texture_to_left"}
```

加载器会自动求逆。RT 支持 JSON、YAML、NPY、NPZ 和 3x4/4x4 TXT；JSON/YAML 可声明 `translation_unit` 为 `m`、`cm` 或 `mm`。内部全部换算为米。

`disparity_offset` 是完整分辨率下视差到无穷远平面的偏移；负视差配置可用 `disparity_sign: -1`。这两个量必须与当前左右校正和视差定义一致。

## 3. 推理

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

默认推理 32 次 FIU，与论文设置一致。每个样本输出：

- `*_disparity.npy/.pfm/.png`：最终视差；
- `*_aligned_depth.npy/.png`：GAM 对齐并转换到左相机的单目深度；
- `*_alignment.json`：`alpha`、`beta`、投影覆盖率和耗时。

如果单目先验已经离线生成，可使用 `--mono-depth "cache/*.npy"`，此时不需要加载 Depth Anything V2 权重。

## 4. 训练

训练清单为 JSONL，每行一个同步样本。参考 `configs/train_manifest.example.jsonl`：

```json
{"left":"left.png","right":"right.png","texture":"rgb.png","disparity":"disp.pfm","calibration":"calibration.json","rt":"rt.json"}
```

`calibration` 和 `rt` 可以逐样本指定，也可以通过命令行全局指定。纹理相机不与左相机共享像素坐标，因此训练裁剪只裁左右红外及 GT，同时修改左相机主点；纹理图保持完整视场，GAM 通过真实投影完成融合。

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

训练脚本使用 AdamW（`beta=(0.9, 0.999)`、weight decay `1e-4`）、多项式衰减、初始 smooth-L1 与迭代 L1 监督。Depth Anything V2 始终冻结。

旧 SpeckleStereo checkpoint 可作为初始化加载，双目主干参数会复用；GEM/FIU 是新参数，必须在三相机数据上训练或微调。推理日志若提示这些参数缺失，不应把随机初始化结果当作论文性能。

## 5. 测试

```powershell
python -m py_compile core/TCAStereo.py core/tca/*.py train_tca.py infer_tca.py
```

单测包含 RT 方向/单位、RANSAC 全局对齐、GEM 梯度、FIU 有效区域和 Depth Anything V2 张量接口。

## 6. 实际采集注意事项

- 三台相机必须硬件同步或达到可忽略的时差；动态场景中的时间偏差无法由 GAM 修复。
- RT 应来自左红外到纹理相机的联合标定，且必须使用当前校正后的左相机内参。
- `baseline`、RT 平移和 GT 深度必须使用一致单位；示例允许以毫米输入，运行时统一为米。
- GAM 依赖初始双目深度产生 RANSAC 对应点。反光、遮挡过多或标定偏差大时，应检查输出 JSON 中的投影覆盖率与 `alpha/beta`。
- 论文报告的指标依赖 SPECK3D/SceneFlow 训练和论文 checkpoint；仅完成结构接入不会自动获得相同精度。
