import numpy as np
import nodes
import torch
from .utils import rescale, grow_and_blur_mask, adjust_to_aspect_ratio, adjust_to_preferred, apply_padding
from scipy.ndimage import binary_closing, binary_fill_holes

class InpaintCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
                "填充遮罩孔洞": ("BOOLEAN", {"default": True}),
                "模糊遮罩像素": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 256.0, "step": 0.1}),
                "混合像素": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 32.0, "step": 0.1}),
                "缩放算法": (["Nearest-最近邻插值-最快", "Bilinear-双线性插值-较快", "Bicubic-双三次插值-中等", 
                           "Bislerp-双样条插值-较慢", "Lanczos-兰索斯插值-最慢", "Box-区域插值-较慢", 
                           "Hamming-汉明插值-中等"], {"default": "Bicubic-双三次插值-中等"}),
                "输出填充": ([8, 16, 32, 64, 128, 256, 512], {"default": 32}),
            }
        }

    CATEGORY = "CR工具"

    RETURN_TYPES = ("缝合器", "IMAGE", "MASK")
    RETURN_NAMES = ("缝合器", "裁剪图像", "裁剪遮罩")

    FUNCTION = "inpaint_crop"

    def inpaint_crop(self, 图像, 遮罩, 填充遮罩孔洞, 模糊遮罩像素, 混合像素, 缩放算法, 输出填充):
        image = 图像
        mask = 遮罩
        
        if image.shape[0] > 1:
            assert "forced size", "Mode must be 'forced size' when input is a batch of images"
        assert image.shape[0] == mask.shape[0], "Batch size of images and masks must be the same"

        result_stitch = {'x': [], 'y': [], 'original_image': [], 'cropped_mask_blend': [], 'rescale_x': [], 'rescale_y': [], 'start_x': [], 'start_y': [], 'initial_width': [], 'initial_height': []}
        results_image = []
        results_mask = []

        batch_size = image.shape[0]
        for b in range(batch_size):
            one_image = image[b].unsqueeze(0)
            one_mask = mask[b].unsqueeze(0)

            stitch, cropped_image, cropped_mask = self.inpaint_crop_single_image(one_image, one_mask, 填充遮罩孔洞, 模糊遮罩像素, 混合像素, 缩放算法, 输出填充)

            for key in result_stitch:
                result_stitch[key].append(stitch[key])
            cropped_image = cropped_image.squeeze(0)
            results_image.append(cropped_image)
            cropped_mask = cropped_mask.squeeze(0)
            results_mask.append(cropped_mask)

        result_image = torch.stack(results_image, dim=0)
        result_mask = torch.stack(results_mask, dim=0)

        return result_stitch, result_image, result_mask
       
    # Parts of this function are from KJNodes: https://github.com/kijai/ComfyUI-KJNodes
    def inpaint_crop_single_image(self, image, mask, fill_mask_holes, blur_mask_pixels, blend_pixels, rescale_algorithm, padding):
        # 提取缩放算法名称
        rescale_algorithm = rescale_algorithm.split('-')[0].lower()
        
        #Validate or initialize mask
        if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                mask = torch.zeros_like(image[:, :, :, 0])
            else:
                assert False, "mask size must match image size"

        # Fill holes if requested
        if fill_mask_holes:
            holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in holemask:
                mask_np = m.numpy()
                binary_mask = mask_np > 0
                struct = np.ones((5, 5))
                closed_mask = binary_closing(binary_mask, structure=struct, border_value=1)
                filled_mask = binary_fill_holes(closed_mask)
                output = filled_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)

        # Grow and blur mask if requested
        if blur_mask_pixels > 0.001:
            mask = grow_and_blur_mask(mask, blur_mask_pixels)

        # Validate or initialize context mask
        context_mask = mask

        # Ensure mask dimensions match image dimensions except channels
        initial_batch, initial_height, initial_width, initial_channels = image.shape
        mask_batch, mask_height, mask_width = mask.shape
        context_mask_batch, context_mask_height, context_mask_width = context_mask.shape
        assert initial_height == mask_height and initial_width == mask_width, "Image and mask dimensions must match"
        assert initial_height == context_mask_height and initial_width == context_mask_width, "Image and context mask dimensions must match"

        # Extend image and masks to turn it into a big square in case the context area would go off bounds
        extend_y = (initial_width + 1) // 2 # Intended, extend height by width (turn into square)
        extend_x = (initial_height + 1) // 2 # Intended, extend width by height (turn into square)
        new_height = initial_height + 2 * extend_y
        new_width = initial_width + 2 * extend_x

        start_y = extend_y
        start_x = extend_x

        available_top = min(start_y, initial_height)
        available_bottom = min(new_height - (start_y + initial_height), initial_height)
        available_left = min(start_x, initial_width)
        available_right = min(new_width - (start_x + initial_width), initial_width)

        new_image = torch.zeros((initial_batch, new_height, new_width, initial_channels), dtype=image.dtype)
        new_image[:, start_y:start_y + initial_height, start_x:start_x + initial_width, :] = image
        # Mirror image so there's no bleeding of black border when using inpaintmodelconditioning
        # Top
        new_image[:, start_y - available_top:start_y, start_x:start_x + initial_width, :] = torch.flip(image[:, :available_top, :, :], [1])
        # Bottom
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x:start_x + initial_width, :] = torch.flip(image[:, -available_bottom:, :, :], [1])
        # Left
        new_image[:, start_y:start_y + initial_height, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y:start_y + initial_height, start_x:start_x + available_left, :], [2])
        # Right
        new_image[:, start_y:start_y + initial_height, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width, :], [2])
        # Top-left corner
        new_image[:, start_y - available_top:start_y, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y:start_y + available_top, start_x:start_x + available_left, :], [1, 2])
        # Top-right corner
        new_image[:, start_y - available_top:start_y, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y:start_y + available_top, start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])
        # Bottom-left corner
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x:start_x + available_left, :], [1, 2])
        # Bottom-right corner
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])

        new_mask = torch.ones((mask_batch, new_height, new_width), dtype=mask.dtype) # assume ones in extended image
        new_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = mask

        blend_mask = torch.zeros((mask_batch, new_height, new_width), dtype=mask.dtype) # assume zeros in extended image
        blend_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = mask
        # Mirror blend mask so there's no bleeding of border when blending
        # Top
        blend_mask[:, start_y - available_top:start_y, start_x:start_x + initial_width] = torch.flip(mask[:, :available_top, :], [1])
        # Bottom
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x:start_x + initial_width] = torch.flip(mask[:, -available_bottom:, :], [1])
        # Left
        blend_mask[:, start_y:start_y + initial_height, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y:start_y + initial_height, start_x:start_x + available_left], [2])
        # Right
        blend_mask[:, start_y:start_y + initial_height, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width], [2])
        # Top-left corner
        blend_mask[:, start_y - available_top:start_y, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y:start_y + available_top, start_x:start_x + available_left], [1, 2])
        # Top-right corner
        blend_mask[:, start_y - available_top:start_y, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y:start_y + available_top, start_x + initial_width - available_right:start_x + initial_width], [1, 2])
        # Bottom-left corner
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x:start_x + available_left], [1, 2])
        # Bottom-right corner
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width], [1, 2])

        new_context_mask = torch.zeros((mask_batch, new_height, new_width), dtype=context_mask.dtype)
        new_context_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = context_mask

        image = new_image
        mask = new_mask
        context_mask = new_context_mask

        original_image = image
        original_mask = mask
        original_width = image.shape[2]
        original_height = image.shape[1]

        # If there are no non-zero indices in the context_mask, adjust context mask to the whole image
        non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)
        if not non_zero_indices[0].size(0):
            context_mask = torch.ones_like(image[:, :, :, 0])
            context_mask = torch.zeros((mask_batch, new_height, new_width), dtype=mask.dtype)
            context_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] += 1.0
            non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)

        # Compute context area from context mask
        y_min = torch.min(non_zero_indices[0]).item()
        y_max = torch.max(non_zero_indices[0]).item()
        x_min = torch.min(non_zero_indices[1]).item()
        x_max = torch.max(non_zero_indices[1]).item()
        height = context_mask.shape[1]
        width = context_mask.shape[2]
        
        # Grow context area if requested
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1
        y_grow = round(blend_pixels**1.5)
        x_grow = round(blend_pixels**1.5)
        y_min = max(y_min - y_grow // 2, 0)
        y_max = min(y_max + y_grow // 2, height - 1)
        x_min = max(x_min - x_grow // 2, 0)
        x_max = min(x_max + x_grow // 2, width - 1)
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1

        # 智能选择尺寸
        aspect_ratio = x_size / y_size
        
        # 预设尺寸列表
        size_presets = [
            (320, 704),   # 竖5:11
            (384, 640),   # 竖5:8
            (448, 576),   # 竖7:9
            (512, 512),   # 方1:1
            (576, 448),   # 横9:7
            (640, 384),   # 横8:5
            (704, 320)    # 横11:5
        ]
        
        # 计算每个预设尺寸的宽高比
        preset_ratios = [w/h for w, h in size_presets]
        
        # 找到最接近的宽高比
        closest_idx = 0
        min_diff = float('inf')
        for i, ratio in enumerate(preset_ratios):
            diff = abs(ratio - aspect_ratio)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        # 获取最接近的尺寸
        force_width, force_height = size_presets[closest_idx]

        # 判断遮罩区域尺寸与目标尺寸的关系
        # 如果遮罩区域尺寸小于目标尺寸，则不缩放
        if x_size <= force_width and y_size <= force_height:
            # 保持原始尺寸，不进行缩放
            effective_upscale_factor_x = 1.0
            effective_upscale_factor_y = 1.0
            rescale_factor = 1.0
            
            # 调整区域为中心，并扩展到目标尺寸（填充黑色背景）
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            # 计算新的边界
            new_x_min = max(0, center_x - force_width // 2)
            new_x_max = min(width - 1, new_x_min + force_width - 1)
            
            # 如果宽度方向需要填充
            if new_x_max - new_x_min + 1 < force_width:
                if new_x_min == 0:
                    new_x_max = min(width - 1, new_x_min + force_width - 1)
                else:
                    new_x_min = max(0, new_x_max - force_width + 1)
            
            new_y_min = max(0, center_y - force_height // 2)
            new_y_max = min(height - 1, new_y_min + force_height - 1)
            
            # 如果高度方向需要填充
            if new_y_max - new_y_min + 1 < force_height:
                if new_y_min == 0:
                    new_y_max = min(height - 1, new_y_min + force_height - 1)
                else:
                    new_y_min = max(0, new_y_max - force_height + 1)
            
            # 更新边界
            x_min, x_max, y_min, y_max = new_x_min, new_x_max, new_y_min, new_y_max
        else:
            # 遮罩区域尺寸大于目标尺寸，保持原有的缩放逻辑
            effective_upscale_factor_x = 1.0
            effective_upscale_factor_y = 1.0

            #Sub case of ranged size.
            min_width = max_width = force_width
            min_height = max_height = force_height

            assert max_width >= min_width, "max_width must be greater than or equal to min_width"
            assert max_height >= min_height, "max_height must be greater than or equal to min_height"
            # Ensure we set an aspect ratio supported by min_width, max_width, min_height, max_height
            current_width = x_max - x_min + 1
            current_height = y_max - y_min + 1
        
            # Calculate aspect ratio of the selected area
            current_aspect_ratio = current_width / current_height

            # Calculate the aspect ratio bounds
            min_aspect_ratio = min_width / max_height
            max_aspect_ratio = max_width / min_height

            # Adjust target width and height based on aspect ratio bounds
            if current_aspect_ratio < min_aspect_ratio:
                # Adjust to meet minimum width constraint
                target_width = min(current_width, min_width)
                target_height = int(target_width / min_aspect_ratio)
                x_min, x_max, y_min, y_max = adjust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height, target_width, target_height)
                x_min, x_max, y_min, y_max = adjust_to_preferred(x_min, x_max, y_min, y_max, width, height, start_x, start_x+initial_width, start_y, start_y+initial_height)
            elif current_aspect_ratio > max_aspect_ratio:
                # Adjust to meet maximum width constraint
                target_height = min(current_height, max_height)
                target_width = int(target_height * max_aspect_ratio)
                x_min, x_max, y_min, y_max = adjust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height, target_width, target_height)
                x_min, x_max, y_min, y_max = adjust_to_preferred(x_min, x_max, y_min, y_max, width, height, start_x, start_x+initial_width, start_y, start_y+initial_height)
            else:
                # Aspect ratio is within bounds, keep the current size
                target_width = current_width
                target_height = current_height

            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1

            # Adjust to min and max sizes
            max_rescale_width = max_width / x_size
            max_rescale_height = max_height / y_size
            max_rescale_factor = min(max_rescale_width, max_rescale_height)
            rescale_factor = max_rescale_factor
            min_rescale_width = min_width / x_size
            min_rescale_height = min_height / y_size
            min_rescale_factor = min(min_rescale_width, min_rescale_height)
            rescale_factor = max(min_rescale_factor, rescale_factor)

            # Upscale image and masks if requested, they will be downsized at stitch phase
            if rescale_factor < 0.999 or rescale_factor > 1.001:
                samples = image            
                samples = samples.movedim(-1, 1)
                width = round(samples.shape[3] * rescale_factor)
                height = round(samples.shape[2] * rescale_factor)
                samples = rescale(samples, width, height, rescale_algorithm)
                effective_upscale_factor_x = float(width)/float(original_width)
                effective_upscale_factor_y = float(height)/float(original_height)
                samples = samples.movedim(1, -1)
                image = samples

                samples = mask
                samples = samples.unsqueeze(1)
                samples = rescale(samples, width, height, "nearest")
                samples = samples.squeeze(1)
                mask = samples

                samples = blend_mask
                samples = samples.unsqueeze(1)
                samples = rescale(samples, width, height, "nearest")
                samples = samples.squeeze(1)
                blend_mask = samples

                # Do math based on min,size instead of min,max to avoid rounding errors
                y_size = y_max - y_min + 1
                x_size = x_max - x_min + 1
                target_x_size = int(x_size * effective_upscale_factor_x)
                target_y_size = int(y_size * effective_upscale_factor_y)

                x_min = round(x_min * effective_upscale_factor_x)
                x_max = x_min + target_x_size
                y_min = round(y_min * effective_upscale_factor_y)
                y_max = y_min + target_y_size

        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1

        # Ensure width and height are within specified bounds, key for ranged and forced size
        if x_size < force_width:
            x_max = min(x_max + (force_width - x_size), width - 1)
        elif x_size > force_width:
            x_max = x_min + force_width - 1

        if y_size < force_height:
            y_max = min(y_max + (force_height - y_size), height - 1)
        elif y_size > force_height:
            y_max = y_min + force_height - 1

        # Recalculate x_size and y_size after adjustments
        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1

        # Pad area (if possible, i.e. if pad is smaller than width/height) to avoid the sampler returning smaller results
        if padding > 1:
            x_min, x_max = apply_padding(x_min, x_max, width, padding)
            y_min, y_max = apply_padding(y_min, y_max, height, padding)

        # Ensure that context area doesn't go outside of the image
        x_min = max(x_min, 0)
        x_max = min(x_max, width - 1)
        y_min = max(y_min, 0)
        y_max = min(y_max, height - 1)

        # Crop the image and the mask, sized context area
        cropped_image = image[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask_blend = blend_mask[:, y_min:y_max+1, x_min:x_max+1]

        # Grow and blur mask for blend if requested
        if blend_pixels > 0.001:
            cropped_mask_blend = grow_and_blur_mask(cropped_mask_blend, blend_pixels)

        # Return stitch (to be consumed by the class below), image, and mask
        stitch = {'x': x_min, 'y': y_min, 'original_image': original_image, 'cropped_mask_blend': cropped_mask_blend, 'rescale_x': effective_upscale_factor_x, 'rescale_y': effective_upscale_factor_y, 'start_x': start_x, 'start_y': start_y, 'initial_width': initial_width, 'initial_height': initial_height}

        return (stitch, cropped_image, cropped_mask)