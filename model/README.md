# add your new model
If you want to add your own model, you can follow these steps:
1. create a file in `model` folder and implement your model.
2. in `model/__init__.py`, append a line which is `from xxx import XX`.
3. in `config/defaults.py`, add your model's config.
4. in  `experiment` folder, create a folder named by your model, and create a `base.yaml` in that folder.
5. run `python train.py  --config-file experiment/yourmodel/base.yaml` in project folder.


# OWM
dataset: Set14
train method: 30000 iterations for upscale factor 4 and 30000 iterations for upscale factor 2

| scale | measure | Refine | RefineOWM | bicubic |  
| --- | --- | --- | --- | --- |
| 2 | PSNR | 33.167 | 33.135 | 30.272 |
| 2 | SSIM | 0.923 | 0.924 | 0.881 |
| 3 | PSNR | 29.015 | 29.348 | 27.599 |
| 3 | SSIM | 0.841 | 0.847 | 0.793 |
| 4 | PSNR | 26.422 | 27.090 | 26.043 |
| 4 | SSIM | 0.755 | 0.774 | 0.721 |

