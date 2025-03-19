from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
import numpy as np


def get_slicers(image_size, patch_size, tile_step_size):
    steps = compute_steps_for_sliding_window(image_size[1:], patch_size, tile_step_size)

    slicers = []
    for d in range(image_size[0]):
        for sx in steps[0]:
            for sy in steps[1]:
                slicers.append(
                    tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), patch_size)]]))

    return slicers


def compute_steps_for_sliding_window(image_size, tile_size, tile_step_size):
    # Calculate the step sizes in voxels for each dimension
    target_step_sizes_in_voxels = np.array(tile_size) * tile_step_size

    # Calculate the number of steps for each dimension
    num_steps = np.ceil((np.array(image_size) - np.array(tile_size)) / target_step_sizes_in_voxels).astype(int) + 1

    # Calculate the actual step sizes for each dimension
    max_step_values = np.array(image_size) - np.array(tile_size)
    actual_step_sizes = np.where(num_steps > 1, max_step_values / (num_steps - 1), np.inf)

    # Generate the steps for each dimension
    steps = [np.round(actual_step_sizes[i] * np.arange(num_steps[i])).astype(int) for i in range(len(tile_size))]

    return steps


def get_slicers(image_size, patch_size, tile_step_size):
    steps = compute_steps_for_sliding_window(image_size, patch_size, tile_step_size)
    slicers = []
    for sx in steps[0]:
        for sy in steps[1]:
            for sz in steps[2]:
                slicers.append(
                    tuple([slice(sx, sx + patch_size[0]), slice(sy, sy + patch_size[1]),
                           slice(sz, sz + patch_size[2])]))  # 只保留三维切片

    return slicers


if __name__ == '__main__':
    image_size = (336, 336, 266)
    patch_size = (112, 112, 112)
    tile_step_size = 0.7
    slicers = get_slicers(image_size, patch_size, tile_step_size)
    for slice in slicers:
        print(slice)
    raw_img = np.random.rand(1, 1, 336, 336, 224)

    # print(len(rs))
    first_slice = slicers[0]
    crop = raw_img[0, 0, first_slice[0], first_slice[1], first_slice[2]]
    print(crop.shape)
    # print(first_slice)
