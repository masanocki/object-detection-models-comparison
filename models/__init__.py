from .base.coco.yolo import run_yolo_coco_videos, run_yolo_coco_images
from .base.coco.ssd import run_ssd_coco_videos, run_ssd_coco_images
from .base.coco.efficientdet import (
    run_efficientdet_coco_videos,
    run_efficientdet_coco_images,
)
from .base.coco.rtdetrv2 import run_rtdetrv2_coco_videos, run_rtdetrv2_coco_images

from .base.custom.yolo import run_yolo_custom_videos, run_yolo_custom_images
from .base.custom.ssd import run_ssd_custom_videos, run_ssd_custom_images
from .base.custom.efficientdet import (
    run_efficientdet_custom_videos,
    run_efficientdet_custom_images,
)
from .base.custom.rtdetrv2 import run_rtdetrv2_custom_videos, run_rtdetrv2_custom_images
