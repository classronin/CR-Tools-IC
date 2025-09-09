import comfy.utils
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, binary_fill_holes, binary_closing


def rescale(samples, width, height, algorithm: str):
    if algorithm == "bislerp":  # convert for compatibility with old workflows
        algorithm = "bicubic"
    algorithm = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
    samples_pil: Image.Image = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
    samples = F.to_tensor(samples_pil).unsqueeze(0)
    return samples


def grow_and_blur_mask(mask, blur_pixels):
    if blur_pixels > 0.001:
        sigma = blur_pixels / 4
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
        out = []
        for m in growmask:
            mask_np = m.numpy()
            kernel_size = math.ceil(sigma * 1.5 + 1)
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            dilated_mask = grey_dilation(mask_np, footprint=kernel)
            output = dilated_mask.astype(np.float32) * 255
            output = torch.from_numpy(output)
            out.append(output)
        mask = torch.stack(out, dim=0)
        mask = torch.clamp(mask, 0.0, 1.0)

        mask_np = mask.numpy()
        filtered_mask = gaussian_filter(mask_np, sigma=sigma)
        mask = torch.from_numpy(filtered_mask)
        mask = torch.clamp(mask, 0.0, 1.0)
    
    return mask


def adjust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height, target_width, target_height):
    # Calculate the current width and height
    current_width = x_max - x_min + 1
    current_height = y_max - y_min + 1

    # Calculate aspect ratios
    aspect_ratio = target_width / target_height
    current_aspect_ratio = current_width / current_height

    if current_aspect_ratio < aspect_ratio:
        # Adjust width to match target aspect ratio
        new_width = int(current_height * aspect_ratio)
        extend_x = (new_width - current_width)
        x_min = max(x_min - extend_x//2, 0)
        x_max = min(x_max + extend_x//2, width - 1)
    else:
        # Adjust height to match target aspect ratio
        new_height = int(current_width / aspect_ratio)
        extend_y = (new_height - current_height)
        y_min = max(y_min - extend_y//2, 0)
        y_max = min(y_max + extend_y//2, height - 1)

    return int(x_min), int(x_max), int(y_min), int(y_max)


def adjust_to_preferred(x_min, x_max, y_min, y_max, width, height, preferred_x_start, preferred_x_end, preferred_y_start, preferred_y_end):
    # Ensure the area is within preferred bounds as much as possible
    if preferred_x_start <= x_min and preferred_x_end >= x_max and preferred_y_start <= y_min and preferred_y_end >= y_max:
        return x_min, x_max, y_min, y_max

    # Shift x_min and x_max to fit within preferred bounds if possible
    if x_max - x_min + 1 <= preferred_x_end - preferred_x_start + 1:
        if x_min < preferred_x_start:
            x_shift = preferred_x_start - x_min
            x_min += x_shift
            x_max += x_shift
        elif x_max > preferred_x_end:
            x_shift = x_max - preferred_x_end
            x_min -= x_shift
            x_max -= x_shift

    # Shift y_min and y_max to fit within preferred bounds if possible
    if y_max - y_min + 1 <= preferred_y_end - preferred_y_start + 1:
        if y_min < preferred_y_start:
            y_shift = preferred_y_start - y_min
            y_min += y_shift
            y_max += y_shift
        elif y_max > preferred_y_end:
            y_shift = y_max - preferred_y_end
            y_min -= y_shift
            y_max -= y_shift

    return int(x_min), int(x_max), int(y_min), int(y_max)


def apply_padding(min_val, max_val, max_boundary, padding):
    # Calculate the midpoint and the original range size
    original_range_size = max_val - min_val + 1
    midpoint = (min_val + max_val) // 2

    # Determine the smallest multiple of padding that is >= original_range_size
    if original_range_size % padding == 0:
        new_range_size = original_range_size
    else:
        new_range_size = (original_range_size // padding + 1) * padding

    # Calculate the new min and max values centered on the midpoint
    new_min_val = max(midpoint - new_range_size // 2, 0)
    new_max_val = new_min_val + new_range_size - 1

    # Ensure the new max doesn't exceed the boundary
    if new_max_val >= max_boundary:
        new_max_val = max_boundary - 1
        new_min_val = max(new_max_val - new_range_size + 1, 0)

    # Ensure the range still ends on a multiple of padding
    # Adjust if the calculated range isn't feasible within the given constraints
    if (new_max_val - new_min_val + 1) != new_range_size:
        new_min_val = max(new_max_val - new_range_size + 1, 0)

    return new_min_val, new_max_val


def composite(destination, source, x, y, mask=None, multiplier=8, resize_source=False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[3], top + source.shape[2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask
        
    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination