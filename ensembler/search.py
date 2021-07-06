import yaml
import os
import itertools
from flatten_dict import flatten, unflatten
import operator
import hashlib
import shutil

description = "Prepare a grid search of hyperparameters."


def add_argparse_args(parser):
    parser.add_argument('in_dir', type=str)
    return parser


def execute(args):

    dict_args = vars(args)
    in_dir = os.path.abspath(dict_args["in_dir"])

    sweep_config_file = os.path.join(in_dir, 'sweep.yaml')

    with open(sweep_config_file, 'r') as f:
        sweep_config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    sweep_config = flatten(sweep_config, reducer='dot')
    sweep_config = dict(
        sorted(sweep_config.items(), key=operator.itemgetter(0)))

    files = [
        os.path.join(in_dir, f) for f in os.listdir(in_dir)
        if os.path.isfile(os.path.join(in_dir, f)) and f != 'sweep.yaml'
    ]

    keys = sweep_config.keys()
    values = sweep_config.values()
    permutations = itertools.product(*values)

    for permutation in permutations:
        configuration = unflatten(dict(zip(keys, permutation)), splitter='dot')
        config_yaml = yaml.dump(configuration)
        config_hash = hashlib.md5(config_yaml.encode()).hexdigest()
        out_dir = os.path.join(in_dir, config_hash)
        if os.path.exists(out_dir):
            continue
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
            f.write(config_yaml)
        for f in files:
            shutil.copy(f, out_dir)
