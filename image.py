import sys

sys.path.insert(0, "sam2")

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from ytools.bench import test_torch_cuda_time
import shutil


def draw_mask(img, mask):
    mask = mask[..., 0:1]
    img[:, :, 0:1] = (
        img[:, :, 0:1] * (1 - mask) + img[:, :, 0:1] * mask * 0.6 + mask * 102
    ).astype(np.uint8)
    return img


def save_masks(
    img,
    masks,
    scores,
    fold="masks",
):
    if os.path.exists(fold):
        shutil.rmtree(fold)
    os.makedirs(fold, exist_ok=True)
    total_mask = np.zeros_like(img)[..., 0:1]
    for i, (mask, score) in enumerate(zip(masks, scores)):
        masked_save_path = os.path.join(fold, f"{i}_{score:.2f}_gen.jpg")
        mask_save_path = os.path.join(fold, f"{i}_{score:.2f}_mask.jpg")
        mask = np.expand_dims(mask.astype(np.uint8), axis=2)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = draw_mask(img_bgr, mask)

        total_mask = total_mask * (1 - mask) + mask
        cv2.imwrite(mask_save_path, mask * 255)
        cv2.imwrite(masked_save_path, img_bgr)

    masked_save_path = os.path.join(fold, f"gen.jpg")
    mask_save_path = os.path.join(fold, f"mask.jpg")
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = draw_mask(img_bgr, mask)
    cv2.imwrite(mask_save_path, total_mask * 255)
    cv2.imwrite(masked_save_path, img_bgr)


device = torch.device("cuda")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# large
sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# # # tiny
# sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_tiny.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# speedup default
# predictor.model.set_runtime_backend(backend="torch")

# predictor.model.set_runtime_backend(
#     backend="onnxruntime",
#     args={
#         "model_paths": [
#             "models/forward_image_opt.onnx",
#         ],
#         "providers": [
#             "TensorrtExecutionProvider",
#             "CUDAExecutionProvider",
#             "CPUExecutionProvider",
#         ],
#     },
# )

# speedup with onnxruntime
predictor.set_runtime_backend(
    backend="onnxruntime",
    args={
        "model_paths": [
            "models/set_image_e2e_opt.onnx",
        ],
        "providers": [
            # "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    },
)

image = Image.open("./sam2/notebooks/images/truck.jpg")
image = np.array(image.convert("RGB"))

input_point = np.array([[500, 375], [502, 375]])
input_label = np.array([1, 1])


@test_torch_cuda_time()
def run(predictor: SAM2ImagePredictor, image, input_point, input_label):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # predictor.set_image(image)
    # masks, scores, logits = predictor.predict(
    #     point_coords=np.array([[575, 750]]),
    #     point_labels=np.array([0]),
    #     box=np.array([425, 600, 700, 875]),
    #     multimask_output=True,
    # )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    return masks, scores


for _ in range(10):
    masks, scores = run(predictor, image, input_point, input_label)
save_masks(image, masks, scores, "data/test_image")


# torch: Large: 0.149s, tiny: 0.050s
# onnxruntime_cuda: Large: 0.113s
# onnxruntime_trt: Large: 0.080s
# onnxruntime_e2e: Large: 0.063s
