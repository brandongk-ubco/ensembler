import albumentations as A


def get_augments(image_height, image_width):

    # preprocessing_transform = A.Compose([
    #    A.FromFloat(dtype='uint8', always_apply=True),
    #    A.Equalize(p=1.0, by_channels=False),
    #    A.ToFloat(always_apply=True),
    # ])
    preprocessing_transform = None

    patch_transform = A.Compose([
        A.PadIfNeeded(min_height=image_height,
                      min_width=image_width,
                      always_apply=True),
        A.RandomCrop(height=image_height, width=image_width, always_apply=True)
    ])

    train_transform = A.Compose([
        A.RandomScale(scale_limit=0.1),
        A.Rotate(limit=10),
        A.HorizontalFlip(p=0.5),
        # A.ElasticTransform(alpha=20, sigma=20 * 0.05, alpha_affine=20 * 0.03),
        A.OneOf(
            [
                A.JpegCompression(quality_lower=90),
                A.Blur(),
                A.GaussNoise(),
                # A.MultiplicativeNoise(),
            ],
            p=0.25),
        A.OneOf([
            A.RandomBrightness(),
            A.RandomBrightnessContrast(),
            A.RandomContrast()
        ],
                p=0.25),
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
                      always_apply=True),
        A.ToFloat(always_apply=True)
    ])

    return (preprocessing_transform, train_transform, patch_transform,
            test_transform)
