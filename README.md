# SFDA
Code release for Domain Adaptation with Source Subject Fusion (SFDA)

## Requirements
- python 3.6+
- PyTorch 1.0

`pip install -r requirements.txt`

## Usage

- download datasets from **[A Dataset for Falling Risk Assessment of the Elderly using Wearable Plantar Pressure(BIBM 2022)](https://doi.org/10.1109/BIBM55620.2022.9995052)** 

- divide datasets to MSST setting as the paper instructed
```
--data
  --P1
    --train
    --test
  --P2
    --train
    --test
  --P3
    --train
    --test
  --P4
    --train
    --test
```
       
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
