RGB_img_res = (3, 192, 256)

dts_type = "nyu"
depth_unit = "cm"
max_depth = 1000.0

augmentation_parameters = {
    "flip": 0.5,
    "mirror": 0.5,
    "c_swap": 0.5,
    "random_crop": 0.5,
    "shifting_strategy": 0.5,
}

MAX_DEPTH_CM = 1000.0
INVALID_DEPTH_THRESHOLD_CM = 1.0
