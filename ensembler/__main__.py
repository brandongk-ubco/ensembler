import ensembler.environment
from ensembler.parameters import args
from ensembler import Tasks
import matplotlib
import pytorch_lightning as pl

matplotlib.use('Agg')

dict_args = vars(args)

pl.seed_everything(dict_args["seed"])

task = Tasks.get(dict_args["task"])
task(args)
