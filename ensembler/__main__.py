import ensembler.environment
from ensembler.parameters import parse_parameters
from ensembler import Tasks
import matplotlib
import pytorch_lightning as pl

matplotlib.use('Agg')

parameters = parse_parameters()

dict_params = vars(parameters)

pl.seed_everything(dict_params["seed"])

task = Tasks.get(dict_params["task"])
task(parameters)
