from ensembler.datasets.helpers import split_dataset
import pandas as pd
import uuid


class TestSpliteDataFrame:
    def test_split_single_class_unevenly(self):
        df = pd.DataFrame([{
            "sample": uuid.uuid4(),
            "background": 0.1,
            "1": 0.9
        }, {
            "sample": uuid.uuid4(),
            "background": 0.1,
            "1": 0.9
        }])

        first_samples, second_samples = split_dataset(df, percent=10.)

        assert len(first_samples) == 0
        assert len(second_samples) == 2

    def test_split_single_class_evenly(self):
        df = pd.DataFrame([{
            "sample": uuid.uuid4(),
            "background": 0.1,
            "1": 0.9
        }, {
            "sample": uuid.uuid4(),
            "background": 0.1,
            "1": 0.9
        }])

        first_samples, second_samples = split_dataset(df, percent=50.)

        assert len(first_samples) == 1
        assert len(second_samples) == 1
