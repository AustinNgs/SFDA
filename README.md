# SFDA
Code release for Domain Adaptation with Source Subject Fusion (SFDA)

## Requirements
- python 3.6+
- PyTorch 1.0

`pip install -r requirements.txt`

## Usage

- download datasets from https://github.com/scut-bds/PlantPre

- divide datasets to MSST setting as the paper instructed

- write your config file

- `python main.py --config /path/to/your/config/yaml/file`

- train

  `python main.py --config pressure-train-config.yaml`

- test

  `python main.py --config pressure-test-config.yaml`
  
- monitor (tensorboard required)

  `tensorboard --logdir .`
  
## Citation
please cite:
```

```

## Contact
- austin_wushibin@163.com
