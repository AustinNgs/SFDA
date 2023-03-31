import yaml
import easydict
from os.path import join
import sys
sys.path[0] = '/home/lab-wu.shibin/SFDA'

class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


import argparse
parser = argparse.ArgumentParser(description='Code for *SFDA*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='pressure-train-config.yaml', help='/home/lab-wu.shibin/SFDA')

args = parser.parse_args()

config_file = args.config

args = yaml.load(open(config_file), Loader=yaml.FullLoader)

save_config = yaml.load(open(config_file), Loader=yaml.FullLoader)

args = easydict.EasyDict(args)

dataset = None

# Modify the dataset name here to follow different MSST settings
if args.data.dataset.name == 'pressure1':  # choices=['pressure1', 'pressure2', 'pressure3', 'pressure4']
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['train', 'test'],
    files=[
        'train/list.txt',
        'test/list.txt'
    ],
    prefix=args.data.dataset.root_path)
    dataset.prefixes = [join(dataset.path, 'train'), join(dataset.path, 'test')]
else:
    raise Exception(f'dataset {args.data.dataset.name} not supported!')

source_domain_name = dataset.domains[args.data.dataset.source]
target_domain_name = dataset.domains[args.data.dataset.target]
source_file = dataset.files[args.data.dataset.source]
target_file = dataset.files[args.data.dataset.target]
