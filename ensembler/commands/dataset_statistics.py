import os
from tqdm import tqdm
import pandas as pd
import json
from ensembler.datasets import split_dataset, Datasets
import torch
from argh import arg


@arg('dataset', choices=Datasets.choices())
def dataset_statistics(data_dir: str,
                       dataset: Datasets,
                       num_workers: int = os.cpu_count() - 1):

    dataset_name = dataset
    dataset = Datasets[dataset]
    outdir = os.path.join(data_dir, dataset_name)
    all_data = dataset.get_all_dataloader(os.path.join(data_dir, dataset_name))

    dataloader = torch.utils.data.DataLoader(all_data,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             shuffle=False,
                                             drop_last=False)

    image_names = dataloader.dataset.get_image_names()
    results = pd.DataFrame(columns=["sample"] + list(dataset.classes),
                           index=range(len(image_names)))

    statistics_file = os.path.join(outdir, "class_samples.csv")

    if os.path.exists(statistics_file):
        results = pd.read_csv(statistics_file)
    else:
        for batch, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            for i in range(x.shape[0]):
                img = x[i, :, :, :]
                mask = y[i, :, :, :]
                pixels = img.size()[1] * img.size()[2]
                class_counts = mask.sum(axis=0).sum(axis=0).numpy() / pixels
                img_results = dict(zip(dataset.classes, class_counts))
                results.iloc[batch + i] = [image_names[batch + i]] + list(
                    img_results.values())
        assert not results.isna().any(axis=None)
        results.to_csv(statistics_file, index=False)

    results.sort_values(by=['sample'], inplace=True)
    num_images = len(results)

    assert num_images == len(image_names)

    test_images, trainval_images = split_dataset(results, 15.)

    samples = {
        "trainval": trainval_images["sample"].tolist(),
        "test": test_images["sample"].tolist()
    }

    with open(os.path.join(outdir, "split.json"), "w") as splitfile:
        json.dump(samples, splitfile)
