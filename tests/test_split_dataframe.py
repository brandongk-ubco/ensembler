from ensembler.utils import split_dataframe
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

        first_samples, second_samples = split_dataframe(df, percent=10.)

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

        first_samples, second_samples = split_dataframe(df, percent=50.)

        assert len(first_samples) == 1
        assert len(second_samples) == 1

    # def test_sample_single_class_one_sample(self):
    #     df = pd.DataFrame([{
    #         "sample": uuid.uuid4(),
    #         "background": 1,
    #         "1": 0
    #     }, {
    #         "sample": uuid.uuid4(),
    #         "background": 0.1,
    #         "1": 1
    #     }])

    #     result = sample_dataframe(df)

    #     assert len(result) == 1
    #     assert result.iloc[0]["sample"] == df.iloc[1]["sample"]

    # def test_sample_single_class_many_samples(self):
    #     df = pd.DataFrame()

    #     for i in range(100):
    #         df = df.append(pd.DataFrame([{
    #             "sample": uuid.uuid4(),
    #             "background": 0.5,
    #             "1": 0.5
    #         }]),
    #                        ignore_index=True)

    #     result = sample_dataframe(df)

    #     assert len(result) == 100

    # def test_large_dataset(self):
    #     sample_file = os.path.join(os.path.dirname(__file__), "fixtures",
    #                                "class_samples.csv")
    #     df = pd.read_csv(sample_file)

    #     result = sample_dataframe(df)

    #     assert len(result) == len(df)
    #     assert len(df[~result.index.isin(df.index)]) == 0
