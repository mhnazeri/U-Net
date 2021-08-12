# U-Net
PyTorch implementation of U-Net.

<p align="center">
  <img src="UNet.svg"  height="300" width="600"/>
</p>


<p align="center">
  <img src="unet_results.svg"  height="400" width="600"/>
</p>

### Dependencies
You can install the project dependencies by running:
```bash
pip install -r requirements
```
For logging we used [wandb](https://wandb.ai), you need to create an account to log the metrics and model output.
Alternatively you can use matplotlib to plot model outputs.

[Cityscapes](https://www.cityscapes-dataset.com/) and [Carla segmentation challenge](https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge) have been used as datasets (Currently cityscapes is not implemented). The dataset files are resides in `data` folder and its structure is as follows:
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

You can download the pretrain model for Carla dataset from [here](https://drive.google.com/file/d/1pJsQuxfl-XettKYkQxuu-j0kyOVVPzvi/view?usp=sharing), and modify the `directory.load` parameter in `config.yaml` to load the pre-trained model.
