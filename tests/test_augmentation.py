import random

import numpy as np

from augmentation import augmentation2D, random_crop
from globals import augmentation_parameters


def test_augmentation_probabilities_default_to_half():
    assert augmentation_parameters == {
        "flip": 0.5,
        "mirror": 0.5,
        "c_swap": 0.5,
        "random_crop": 0.5,
        "shifting_strategy": 0.5,
    }


def test_flip_augmentation_reverses_vertical_axis(monkeypatch):
    img = np.arange(2 * 3 * 1).reshape(2, 3, 1)
    depth = np.arange(2 * 3 * 1).reshape(2, 3, 1) + 10

    values = iter([0.0, 1.0, 1.0, 1.0, 1.0])
    monkeypatch.setattr(random, "uniform", lambda *_args: next(values))
    monkeypatch.setattr(random, "random", lambda: 1.0)

    augmented_img, augmented_depth = augmentation2D(img, depth, print_info_aug=False)

    np.testing.assert_array_equal(augmented_img, img[::-1, :, :])
    np.testing.assert_array_equal(augmented_depth, depth[::-1, :, :])


def test_random_crop_can_select_bottom_right_valid_crop(monkeypatch):
    img = np.arange(5 * 6 * 1).reshape(5, 6, 1)
    depth = np.arange(5 * 6 * 1).reshape(5, 6, 1) + 100
    randint_calls = []

    def choose_upper_bound(upper_bound):
        randint_calls.append(upper_bound)
        return upper_bound - 1

    monkeypatch.setattr(np.random, "randint", choose_upper_bound)

    cropped_img, cropped_depth = random_crop(img, depth, crop_size=(3, 4))

    assert randint_calls == [3, 3]
    np.testing.assert_array_equal(cropped_img, img[2:5, 2:6, :])
    np.testing.assert_array_equal(cropped_depth, depth[2:5, 2:6, :])
