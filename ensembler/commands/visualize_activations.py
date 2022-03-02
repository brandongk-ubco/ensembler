from ensembler.activations import PWLinear, Cos
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import os
import torch

sns.set(style="whitegrid")
matplotlib.use('Agg')


def visualize_activations(base_dir: str):
    t = torch.linspace(-2, 2, 100)

    result = PWLinear()(t)

    x = t.numpy()
    y = result.numpy()

    plot = sns.lineplot(x=x, y=y)
    fig = plot.get_figure()

    outfile = os.path.join(base_dir, "piecewise_linear.png")

    fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    t = torch.linspace(-4, 6, 100)
    result = Cos()(t)

    x = t.numpy()
    y = result.numpy()

    plot = sns.lineplot(x=x, y=y)
    fig = plot.get_figure()

    outfile = os.path.join(base_dir, "cos.png")

    fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
