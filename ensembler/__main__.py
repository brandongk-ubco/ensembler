import ensembler.environment
from ensembler.parameters import args
from ensembler import Tasks
import matplotlib
import pytorch_lightning as pl

pl.seed_everything(42)
matplotlib.use('Agg')

dict_args = vars(args)

task = Tasks.get(dict_args["task"])
task(args)
