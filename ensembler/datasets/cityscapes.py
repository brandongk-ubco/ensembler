from ensembler.datasets.CompressedNpzDataset import CompressedNpzDataset


class CityscapesDataset(CompressedNpzDataset):
    classes = [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train", "motorcycle",
        "bicycle"
    ]

    num_classes = len(classes)
    loss_weights = [1.] * num_classes
    num_channels = 3
