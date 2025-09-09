import math
import torch
from .utils import rescale, composite


class InpaintStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "缝合器": ("缝合器",),
                "重绘图像": ("IMAGE",),
                "缩放算法": (["Nearest-最近邻插值-最快", "Bilinear-双线性插值-较快", "Bicubic-双三次插值-中等", 
                           "Bislerp-双样条插值-较慢", "Lanczos-兰索斯插值-最慢", "Box-区域插值-较慢", 
                           "Hamming-汉明插值-中等"], {"default": "Bicubic-双三次插值-中等"}),
            }
        }

    CATEGORY = "CR工具"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)

    FUNCTION = "inpaint_stitch"

    def inpaint_stitch(self, 缝合器, 重绘图像, 缩放算法):
        inpainted_image = 重绘图像
        # 提取缩放算法名称
        rescale_algorithm = 缩放算法.split('-')[0].lower()
        
        results = []

        batch_size = inpainted_image.shape[0]
        assert len(缝合器['x']) == batch_size, "Stitch size doesn't match image batch size"
        for b in range(batch_size):
            one_image = inpainted_image[b]
            one_stitch = {}
            for key in 缝合器:
                # Extract the value at the specified index and assign it to the single_stitch dictionary
                one_stitch[key] = 缝合器[key][b]
            one_image = one_image.unsqueeze(0)
            one_image, = self.inpaint_stitch_single_image(one_stitch, one_image, rescale_algorithm)
            one_image = one_image.squeeze(0)
            results.append(one_image)

        # Stack the results to form a batch
        result_batch = torch.stack(results, dim=0)

        return (result_batch,)

    def inpaint_stitch_single_image(self, stitch, inpainted_image, rescale_algorithm):
        original_image = stitch['original_image']
        cropped_mask_blend = stitch['cropped_mask_blend']
        x = stitch['x']
        y = stitch['y']
        stitched_image = original_image.clone().movedim(-1, 1)
        start_x = stitch['start_x']
        start_y = stitch['start_y']
        initial_width = stitch['initial_width']
        initial_height = stitch['initial_height']

        inpaint_width = inpainted_image.shape[2]
        inpaint_height = inpainted_image.shape[1]

        # Downscale inpainted before stitching if we upscaled it before
        if stitch['rescale_x'] < 0.999 or stitch['rescale_x'] > 1.001 or stitch['rescale_y'] < 0.999 or stitch['rescale_y'] > 1.001:
            samples = inpainted_image.movedim(-1, 1)

            width = math.ceil(float(inpaint_width)/stitch['rescale_x'])+1
            height = math.ceil(float(inpaint_height)/stitch['rescale_y'])+1
            x = math.floor(float(x)/stitch['rescale_x'])
            y = math.floor(float(y)/stitch['rescale_y'])

            samples = rescale(samples, width, height, rescale_algorithm)
            inpainted_image = samples.movedim(1, -1)
            
            samples = cropped_mask_blend.movedim(-1, 1)
            samples = samples.unsqueeze(0)
            samples = rescale(samples, width, height, rescale_algorithm)
            samples = samples.squeeze(0)
            cropped_mask_blend = samples.movedim(1, -1)
            cropped_mask_blend = torch.clamp(cropped_mask_blend, 0.0, 1.0)

        output = composite(stitched_image, inpainted_image.movedim(-1, 1), x, y, cropped_mask_blend, 1).movedim(1, -1)

        # Crop out from the extended dimensions back to original.
        cropped_output = output[:, start_y:start_y + initial_height, start_x:start_x + initial_width, :]
        output = cropped_output
        return (output,)