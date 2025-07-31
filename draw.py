import sys

sys.path.insert(0, "sam2")
import os
import numpy as np
import cv2
import shutil


def gen_image_writer(fold):
    os.makedirs(fold, exist_ok=True)

    def writer(image):
        if image is None:
            return
        cv2.imwrite(os.path.join(fold, "gen.jpg"), image)

    return writer


def gen_video_writer(fold, frame_width, frame_height, fps=30):
    os.makedirs(fold, exist_ok=True)
    video_writer = cv2.VideoWriter(
        os.path.join(fold, "gen.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    def writer(image):
        if image is None:
            video_writer.release()
            return
        video_writer.write(image)

    return writer


def draw_mask(img, mask, id=0):
    mask = mask[..., 0:1]
    img[:, :, id : id + 1] = (
        img[:, :, id : id + 1] * (1 - mask)
        + img[:, :, id : id + 1] * mask * 0.6
        + mask * 102
    ).astype(np.uint8)
    return img


def save_masks(
    img,
    masks,
    fold="masks",
    deal_func=None,
):
    if deal_func is None:
        deal_func = gen_image_writer(fold)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if masks is not None:
        for i, mask in enumerate(masks):
            if mask is None:
                continue
            mask = np.expand_dims(mask.astype(np.uint8), axis=2)
            img_bgr = draw_mask(img_bgr, mask, id=i)
    deal_func(img_bgr)
