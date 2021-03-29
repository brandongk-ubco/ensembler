import albumentations as A


def get_augments(image_height, image_width):
    train_transform = A.Compose([
        A.PadIfNeeded(min_height=image_height,
                      min_width=image_width,
                      always_apply=True),
        A.RandomCrop(height=image_height, width=image_width, always_apply=True)
    ])

    val_transform = A.Compose([
        A.PadIfNeeded(min_height=image_height,
                      min_width=image_width,
                      always_apply=True),
        A.RandomCrop(height=image_height, width=image_width, always_apply=True)
    ])

    test_transform = A.Compose([
        A.PadIfNeeded(min_height=None,
                      min_width=None,
                      pad_height_divisor=512,
                      pad_width_divisor=512,
                      always_apply=True)
    ])

    return (train_transform, val_transform, test_transform)
