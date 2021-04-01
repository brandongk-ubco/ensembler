import albumentations as A


def get_augments(image_height, image_width):
    patch_transform = A.Compose([
        A.PadIfNeeded(min_height=image_height,
                      min_width=image_width,
                      always_apply=True),
        A.RandomCrop(height=image_height, width=image_width, always_apply=True)
    ])

    train_transform = A.Compose([
        A.FromFloat(dtype='uint8', always_apply=True),
        A.OneOf([
            A.CoarseDropout(),
            A.GaussNoise(),
            A.ISONoise(),
            A.MultiplicativeNoise(),
            A.RandomBrightness(),
            A.RandomBrightnessContrast(),
            A.RandomContrast()
        ]),
        A.ToFloat(always_apply=True),
    ])

    test_transform = A.Compose([
        A.PadIfNeeded(min_height=None,
                      min_width=None,
                      pad_height_divisor=256,
                      pad_width_divisor=256,
                      border_mode=0,
                      value=0.,
                      mask_value=0.,
                      always_apply=True)
    ])

    return (train_transform, patch_transform, test_transform)
