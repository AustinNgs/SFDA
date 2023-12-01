# SFDA: Domain Adaptation with Source Subject Fusion Based On Multi-source And Single-target Fall Risk Assessment
- Code release for Domain Adaptation with Source Subject Fusion (SFDA)
- Users could implement their own project by code here but do not forget to cite

## Requirements
- python 3.6+
- PyTorch 1.4
- CUDA 10.2

`pip install -r requirements.txt`

## Usage

- download datasets from **[A Dataset for Falling Risk Assessment of the Elderly using Wearable Plantar Pressure(BIBM 2022)](https://doi.org/10.1109/BIBM55620.2022.9995052)** 

- divide datasets to MSST setting as the paper instructed
```
--data
  --pressure1
    --train/images
    --test/images

  --pressure2
    --train/images
    --test/images

  --pressure3
    --train/images
    --test/images

  --pressure4
    --train/images
    --test/images

```
       
- write your config file

- `python main.py --config /path/to/your/config/yaml/file`

- train

  `python main.py --config pressure-train-config.yaml`

- test

  `python main.py --config pressure-test-config.yaml`
  
- monitor (tensorboard required)

  `tensorboard --logdir .`

## Checkpoint
- Users could reproduce the results in **[Our paper](https://doi.org/10.1109/TNSRE.2023.3337861)** by saved model from **[Baidu Netdisk](https://pan.baidu.com/s/1PIQNwAYq7nLStmo0TVncDA)** with password: sfda
 
 
## Citation
please cite:
```
@ARTICLE{10335742,
  author={Wu, Shibin and Shu, Lin and Song, Zhen and Xu, Xiangmin},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  title={SFDA: Domain Adaptation with Source Subject Fusion Based On Multi-source And Single-target Fall Risk Assessment}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TNSRE.2023.3337861}}
```

## Contact
- austin_wushibin@163.com
