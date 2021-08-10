import argh
from ensembler.commands import dataset_initialize, dataset_statistics, evaluate

parser = argh.ArghParser()
parser.add_commands([dataset_initialize, dataset_statistics, evaluate])

if __name__ == '__main__':
    parser.dispatch()
