import ensembler.environment
from ensembler.parameters import parse_parameters
from ensembler import Tasks

parameters = parse_parameters()

dict_params = vars(parameters)

task = Tasks.get(dict_params["task"])
task(parameters)
