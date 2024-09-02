## Introduction

Pytorch implementation for [Unifying Spike Perception and Prediction: A Compact Spike Representation Model using Multi-scale Correlation](https://openreview.net/forum?id=lkOB0hBLS5), ACM Multimedia 2024.

## Prerequisites

- Python 3.8.16 and Conda 4.12.0

- CUDA 10.1

- Environment

  ```python
  conda create -n $YOUR_PY38_ENV_NAME python=3.8.16
  conda activate $YOUR_PY38_ENV_NAME
  pip install -r requirements.txt
  ```


## Evaluation

```
python eval_mstau.py --spk_path_file dataset\eval_spk_path.txt --half_win 6 --input_len 10 --batch 2 --worker_num 2 --device YOUR_DEVICE --load_extractor_checkpoint checkpoint\extractor.pth --load_predictor_checkpoint checkpoint\predictor.pth --load_compositor_checkpoint checkpoint\compositor.pth
```

## Citation

If you find this work useful for your research, please cite:

```
@inproceedings{feng2024unifying,
  title={Unifying Spike Perception and Prediction: A Compact Spike Representation Model using Multi-scale Correlation},
  author={Feng, Kexiang and Jia, Chuanmin and Ma, Siwei and Gao, Wen},
  booktitle={ACM Multimedia 2024}
}
```

