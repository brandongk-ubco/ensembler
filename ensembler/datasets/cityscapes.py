from ensembler.datasets._base import base_get_dataloaders, base_get_all_dataloader
from ensembler.datasets.CompressedNpzDataset import CompressedNpzDataset
from functools import partial
import torch

classes = [
    "background", "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person",
    "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

num_classes = len(classes)
loss_weights = [1.] * num_classes
loss_weights[0] = 0.
num_channels = 3


class CityscapesDataset(CompressedNpzDataset):
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name, image, mask = super().load_image(self.images[idx])
        image = image.transpose(1, 2, 0)
        mask = mask.transpose(1, 2, 0)
        return (image, mask)


get_dataloaders = partial(base_get_dataloaders, Dataset=CityscapesDataset)
get_all_dataloader = partial(base_get_all_dataloader,
                             Dataset=CityscapesDataset)
