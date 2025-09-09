from .inpaint_crop import InpaintCrop
from .inpaint_stitch import InpaintStitch

NODE_CLASS_MAPPINGS = {
    "InpaintCrop": InpaintCrop,
    "InpaintStitch": InpaintStitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCrop": "局部重绘(裁剪)",
    "InpaintStitch": "局部重绘(拼接)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']