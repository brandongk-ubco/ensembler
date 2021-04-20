import os
from ensembler.p_tqdm import t_imap as mapper
import numpy as np
from ensembler.datasets import Datasets
from functools import partial
import gzip

description = "Prepare dataset statistics required for sampling."


def add_argparse_args(parser):
    pass


def get_image(dataloader, idx):
    return idx, dataloader[idx]


def execute(args):

    dict_args = vars(args)

    indir = os.path.join(dict_args["data_dir"], "unprocessed",
                         dict_args["dataset_name"])

    intializer = Datasets.get_initializer(dict_args["dataset_name"])
    all_data = intializer.get_all_dataloader(indir)

    image_names = all_data.get_image_names()

    outdir = os.path.join(dict_args["data_dir"], dict_args["dataset_name"])

    num_images = len(all_data)
    os.makedirs(outdir, exist_ok=True)

    for idx, (image, mask) in mapper(partial(get_image, all_data),
                                     range(num_images),
                                     total=num_images):
        image_name = image_names[idx]
        outfile = os.path.join(outdir, image_name)
        np.savez_compressed(file=outfile, image=image, mask=mask)
