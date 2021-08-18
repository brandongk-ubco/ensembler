import argh
from ensembler.commands import dataset_initialize, dataset_statistics, evaluate, combine_metrics

parser = argh.ArghParser()
parser.add_commands(
    [dataset_initialize, dataset_statistics, evaluate, combine_metrics])

if __name__ == '__main__':
    parser.dispatch()
