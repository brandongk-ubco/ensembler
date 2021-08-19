import argh
from ensembler.commands import dataset_initialize, dataset_statistics, evaluate, combine_metrics, evaluate_diversity

parser = argh.ArghParser()
parser.add_commands([
    dataset_initialize, dataset_statistics, evaluate, combine_metrics,
    evaluate_diversity
])

if __name__ == '__main__':
    parser.dispatch()
