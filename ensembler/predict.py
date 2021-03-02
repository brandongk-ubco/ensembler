import pytorch_lightning as pl
import matplotlib
from Model import Segmenter
import numpy as np
import os
from train import get_augments
from tqdm import tqdm
from datasets import Datasets
from parameters import args

matplotlib.use('Agg')

model = "/mnt/d/work/repos/ensembler/lightning_logs/version_2/checkpoints/epoch=55-step=37687.ckpt"
outdir = "/mnt/d/work/repos/ensembler/lightning_logs/version_2/predictions/"

os.makedirs(outdir, exist_ok=True)


def predict(i, x, y):
    x = x.to("cuda")
    y = y.to("cuda")
    batch_size = x.shape[0]
    y_hat = m(x)
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    assert y_hat.shape == y.shape

    for img_num in range(batch_size):
        name = image_names[i * batch_size + img_num]
        prediction = y_hat[img_num, :, :, :]
        img = x[img_num, :, :, :]
        mask = y[img_num, :, :, :]

        np.savez_compressed(os.path.join(outdir, name),
                            prediction=prediction,
                            image=img,
                            mask=mask)


if __name__ == '__main__':

    dict_args = vars(args)

    dataset = Datasets.get(dict_args["dataset"])

    m = Segmenter.load_from_checkpoint(model,
                                       get_augments=get_augments,
                                       **dict_args)

    test_dataloader = m.test_dataloader()
    image_names = test_dataloader.dataset.dataset.get_image_names()

    m = m.to("cuda")

    m.eval()
    m.freeze()
    for i, (x, y) in tqdm(enumerate(test_dataloader),
                          total=len(test_dataloader)):
        predict(i, x, y)
