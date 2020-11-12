import numpy as np


class AugmentedDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset.__getitem__(idx)
        image = np.array(image)
        mask = np.array(mask)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        mask = mask.unsqueeze(0)

        image = image.float()
        mask = mask.float()
        if image.size()[1:] != mask.size()[1:]:
            import pdb
            pdb.set_trace()

        return image, mask
