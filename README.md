# DCGAN MedNIST Hand X-ray

This project modularizes the Github Project: https://github.com/MatteoLai/dcgan-xray-generation-tutorial

The project deliberately keeps the original tutorial's core implementation as unchanged as practical:

- MedNIST dataset loaded with MONAI
- only the `Hand` class is used
- image size: `1 x 64 x 64`
- `ScaleIntensityRanged`: `[0,255] -> [-1,1]`
- `Rotate90d(k=3)`
- original DCGAN Generator channel progression
- original DCGAN Discriminator channel progression
- `Sigmoid + binary_cross_entropy`
- Adam with `lr=0.0002`, `beta1=0.5`, `beta2=0.999`
- PyTorch Lightning manual optimization
- 100 epochs by default


## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Train

```bash
python main.py --mode train
```

The MedNIST dataset is downloaded automatically by MONAI.

Default settings:

```text
epochs       = 100
batch_size   = 128
latent_dim   = 100
lr           = 0.0002
beta1        = 0.5
beta2        = 0.999
```

Generated images are written to:

```text
results/generated_images/
```

The loss curve is written to:

```text
results/loss_curve.png
```

Checkpoints are written to:

```text
checkpoints/
```

## 3. Test / Generate synthetic hand X-rays

```bash
python main.py --mode test
```

This loads:

```text
checkpoints/generator_final.pth
```

and saves generated images to:

```text
results/test_results/
```

## Notes

`test.py` takes random latent vectors as input because this is an unconditional GAN.
It does not take a real hand X-ray image as an input.
