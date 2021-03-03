import pandas as pd


def RoundUp(x, mul):
    return ((x + mul - 1) & (-mul))


def split_dataframe(dataframe, percent, seed=42):

    num_total = len(dataframe)

    num_first_samples = round(num_total * percent / 100.)
    num_second_samples = num_total - num_first_samples

    print("Splitting {} samples into {} and {}".format(num_total,
                                                       num_first_samples,
                                                       num_second_samples))

    first_samples = pd.DataFrame(columns=dataframe.columns,
                                 index=range(num_first_samples))
    second_samples = pd.DataFrame(columns=dataframe.columns,
                                  index=range(num_second_samples))
    class_counts = dataframe.astype(bool).sum(axis=0)[1:-1]
    class_counts.sort_values(inplace=True)

    first_idx = 0
    second_idx = 0
    available_samples = dataframe.copy()
    for i, clazz in enumerate(class_counts.index):
        clazz_df = available_samples[available_samples[clazz] > 0]
        available_class_count = len(clazz_df)

        if i == len(class_counts.index) - 1:
            num_class_first_samples = num_first_samples - first_idx
            num_class_second_samples = available_class_count - num_class_first_samples
        else:
            num_class_first_samples = round(available_class_count * percent /
                                            100.)
            num_class_second_samples = available_class_count - num_class_first_samples
        print("Splitting {} class {} samples into {} and {}".format(
            available_class_count, clazz, num_class_second_samples,
            num_class_first_samples))

        clazz_first_samples = clazz_df.sample(n=num_class_first_samples,
                                              random_state=seed)
        clazz_second_samples = clazz_df[~clazz_df.index.
                                        isin(clazz_first_samples.index)]

        second_samples[second_idx:second_idx +
                       num_class_second_samples] = dataframe[
                           dataframe.index.isin(clazz_second_samples.index)]

        first_samples[first_idx:first_idx +
                      num_class_first_samples] = dataframe[
                          dataframe.index.isin(clazz_first_samples.index)]

        available_samples = available_samples[~available_samples.index.
                                              isin(clazz_df.index)]

        first_idx += num_class_first_samples
        second_idx += num_class_second_samples

        assert second_idx == len(second_samples.dropna())
        assert first_idx == len(first_samples.dropna())

        print("Allocated {} images, {} remaining.".format(
            first_idx + second_idx, len(available_samples)))

    assert len(available_samples) == 0
    assert first_idx == num_first_samples
    assert second_idx == num_second_samples

    return first_samples, second_samples