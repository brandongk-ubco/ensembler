import os
from ensembler.p_tqdm import t_imap as mapper
import numpy as np
from ensembler.datasets import Datasets
from functools import partial

description = "Prepare dataset statistics required for sampling."


def get_image(dataloader, idx):
    return idx, dataloader[idx]


def dataset_initialize(data_dir: str, dataset_name: str):

    indir = os.path.join(data_dir, "unprocessed", dataset_name)

    intializer = Datasets.get_initializer(dataset_name)
    all_data = intializer.get_all_dataloader(indir)

    image_names = all_data.get_image_names()

    outdir = os.path.join(data_dir, dataset_name)

    num_images = len(all_data)
    os.makedirs(outdir, exist_ok=True)

    for idx, (image, mask) in mapper(partial(get_image, all_data),
                                     range(num_images),
                                     total=num_images):
        image_name = image_names[idx]
        outfile = os.path.join(outdir, image_name)
        np.savez_compressed(file=outfile, image=image, mask=mask)
