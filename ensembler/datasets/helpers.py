import pandas as pd


def split_dataset(dataframe, percent, seed=42):

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
    class_counts = dataframe.astype(bool).sum(axis=0)[2:]
    class_counts = class_counts.sort_values(ascending=True)

    first_idx = 0
    second_idx = 0
    available_samples = dataframe.copy()
    for i, clazz in enumerate(class_counts.index):
        clazz_df = available_samples[available_samples[clazz] > 0]
        available_class_count = len(clazz_df)

        num_class_first_samples = round(available_class_count * percent / 100.)
        num_class_second_samples = available_class_count - num_class_first_samples

        num_class_first_samples = min(num_class_first_samples,
                                      len(first_samples) - first_idx)
        num_class_second_samples = min(num_class_second_samples,
                                       len(second_samples) - second_idx)

        print("Splitting {} class {} samples into {} and {}".format(
            available_class_count, clazz, num_class_second_samples,
            num_class_first_samples))

        clazz_first_samples = clazz_df.sample(n=num_class_first_samples,
                                              random_state=seed)
        clazz_second_samples = clazz_df[~clazz_df.index.
                                        isin(clazz_first_samples.index)]
        if len(clazz_second_samples) > num_class_second_samples:
            clazz_second_samples = clazz_second_samples.sample(
                n=num_class_second_samples, random_state=seed)

        new_second_samples = dataframe[dataframe.index.isin(
            clazz_second_samples.index)]

        second_sample_count = len(new_second_samples)

        second_sample_rows = second_samples.iloc[second_idx:second_idx +
                                                 second_sample_count]

        new_second_samples.index = second_sample_rows.index

        second_samples.iloc[second_idx:second_idx +
                            second_sample_count] = new_second_samples

        new_first_samples = dataframe[dataframe.index.isin(
            clazz_first_samples.index)]

        first_sample_count = len(new_first_samples)
        first_sample_rows = first_samples.iloc[first_idx:first_idx +
                                               first_sample_count]
        new_first_samples.index = first_sample_rows.index

        first_samples.iloc[first_idx:first_idx +
                           first_sample_count] = new_first_samples

        available_samples = available_samples[~available_samples.index.
                                              isin(clazz_df.index)]

        first_idx += first_sample_count
        second_idx += second_sample_count

        assert second_idx == len(second_samples.dropna())
        assert first_idx == len(first_samples.dropna())

        print("Allocated {} images, {} remaining.".format(
            first_idx + second_idx, len(available_samples)))

    assert len(available_samples) == 0
    first_samples = first_samples.dropna()
    second_samples = second_samples.dropna()
    return first_samples, second_samples


def sample_dataset(dataframe, seed=42):
    class_counts = dataframe.astype(bool).sum(axis=0)[2:]
    class_counts = class_counts.sort_values(ascending=True)
    sample_count = class_counts.min()

    print("Sampling {} from each class.".format(sample_count))

    samples = pd.DataFrame(columns=dataframe.columns,
                           index=range(sample_count * len(class_counts)))

    available_samples = dataframe.copy()
    sample_idx = 0

    for i, clazz in enumerate(class_counts.index):
        already_in_dataframe = len(samples[samples[clazz] > 0])
        class_sample_count = sample_count - already_in_dataframe
        if class_sample_count <= 0:
            continue
        clazz_df = available_samples[available_samples[clazz] > 0]
        if len(clazz_df) <= class_sample_count:
            sampled = clazz_df
        else:
            sampled = clazz_df.sample(n=class_sample_count, random_state=seed)

        sample_rows = dataframe[dataframe.index.isin(sampled.index)].copy()
        sampled_count = len(sample_rows)
        target_samples = samples.iloc[sample_idx:sample_idx + sampled_count]
        sample_rows.index = target_samples.index

        samples.iloc[sample_idx:sample_idx + sampled_count] = sample_rows
        available_samples = available_samples[~available_samples.index.
                                              isin(sampled.index)]

        sample_idx += sampled_count

    samples = samples.dropna()

    return samples


def process_split(sample_split, statistics_file):

    test_images = sample_split["test"]
    trainval_images = sample_split["trainval"]

    dataset_df = pd.read_csv(statistics_file)
    test_df = dataset_df[dataset_df["sample"].isin(test_images)]
    trainval_df = dataset_df[dataset_df["sample"].isin(trainval_images)]

    assert len(trainval_images) == len(trainval_df)
    assert len(test_images) == len(test_df)

    #trainval_df = sample_dataset(trainval_df)
    val_df, train_df = split_dataset(trainval_df, 10.)

    val_counts = val_df.astype(bool).sum(axis=0)[2:]
    train_counts = train_df.astype(bool).sum(axis=0)[2:]
    test_counts = test_df.astype(bool).sum(axis=0)[2:]

    print("Train Class Count: {}".format(train_counts))
    print("Validation Class Count: {}".format(val_counts))
    print("Test Class Count: {}".format(test_counts))

    val_images = val_df["sample"].tolist()
    train_images = train_df["sample"].tolist()
    return train_images, val_images, test_images
