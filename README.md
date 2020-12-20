#U-Net

PyTorch implementation of U-Net.

### Dependencies
You can install the project dependencies by running:
```bash
pip install -r requirements
```
For logging we used [wandb](https://wandb.ai), you need to create an account to log the metrics and model output.
Alternatively you can use matplotlib to plot model outputs.

[Cityscapes]() and Carla image segmentation have been used as datasets (Currently cityscapes is not implemented). The dataset files are resides in `data` folder and its structure is as follows:
```bash
├──  carla
│  ├──  dataA
│  │  ├──  CameraRGB
│  │  │  └──  02_00_000.png ...
│  │  └──  CameraSeg
│  │     └──  02_00_000.png ...
│  ├──  DataB
│  ├──  DataC
│  ├──  DataD
│  └──  DataE
└──  cityscapes

```

After editing the config file, you can train the model simply by executing:
```bash
python train.py
```