# SFDA: Domain Adaptation with Source Subject Fusion Based On Multi-source And Single-target Fall Risk Assessment
Code release for Domain Adaptation with Source Subject Fusion (SFDA)

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
      --subject3
      ...
      --subject50
      --list.txt
    --test/images
      --subject1
      ...
      --subject46
      --list.txt
  --pressure2
    --train/images
      --subject1
      ...
      --subject48
      --list.txt
    --test/images
      --subject3
      ...
      --subject50
      --list.txt
  --pressure3
    --train/images
      --subject1
      ...
      --subject50
      --list.txt
    --test/images
      --subject6
      ...
      --subject48
      --list.txt
  --pressure4
    --train/images
      --subject1
      ...
      --subject50
      --list.txt
    --test/images
      --subject10
      ...
      --subject47
      --list.txt
```
       
- write your config file

- `python main.py --config /path/to/your/config/yaml/file`

- train

  `python main.py --config pressure-train-config.yaml`

- test

  `python main.py --config pressure-test-config.yaml`
  
- monitor (tensorboard required)

  `tensorboard --logdir .`

## Saved model
- Users could reproduce the results in **[Our paper](https://doi.org/10.1109/BIBM55620.2022.9995052)** by saved model from **[Baidu Netdisk](https://doi.org/10.1109/BIBM55620.2022.9995052)** 
  
## Citation
please cite:
```

```

## Contact
- austin_wushibin@163.com
