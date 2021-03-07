import pytorch_lightning as pl
import matplotlib
from Model import Segmenter as model
import os
from train import get_augments
from tqdm import tqdm
from datasets import Datasets
from parameters import args
import pandas as pd
import json
from utils import split_dataframe

matplotlib.use('Agg')

if __name__ == '__main__':

    dict_args = vars(args)

    outdir = os.path.join(dict_args["data_dir"], dict_args["dataset"])

    dataset = Datasets.get(dict_args["dataset"])

    m = model(get_augments, **dict_args)

    dataloader = m.all_dataloader()

    image_names = dataloader.dataset.get_image_names()
    results = pd.DataFrame(columns=["sample"] + list(dataset.classes),
                           index=range(len(image_names)))
    batch_size = dict_args["batch_size"]

    statistics_file = os.path.join(outdir, "class_samples.csv")

    if os.path.exists(statistics_file):
        results = pd.read_csv(statistics_file)
    else:
        for batch, (x, y) in tqdm(enumerate(dataloader),
                                  total=len(dataloader)):
            for i in range(x.shape[0]):
                img = x[i, :, :, :]
                mask = y[i, :, :, :]
                pixels = img.size()[1] * img.size()[2]
                class_counts = mask.sum(axis=0).sum(axis=0).numpy() / pixels
                img_results = dict(zip(dataset.classes, class_counts))
                results.iloc[batch * batch_size +
                             i] = [image_names[batch * batch_size + i]] + list(
                                 img_results.values())
        assert not results.isna().any(axis=None)
        results.to_csv(statistics_file, index=False)

    results.sort_values(by=['sample'], inplace=True)
    num_images = len(results)

    assert num_images == len(image_names)

    test_images, trainval_images = split_dataframe(results, 15.)

    samples = {
        "trainval": trainval_images["sample"].tolist(),
        "test": test_images["sample"].tolist()
    }

    with open(os.path.join(outdir, "split.json"), "w") as splitfile:
        json.dump(samples, splitfile)
