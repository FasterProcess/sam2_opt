import threading

import sys

sys.path.insert(0, "sam2")

import os
import shutil
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor
from draw import save_masks, gen_video_writer

# init predictor
sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(
    model_cfg, sam2_checkpoint, device=torch.device("cuda")
)  # type:SAM2VideoPredictor

predictor.speedup()

# test inputs
video_path = "./sam2/notebooks/videos/bedroom.mp4"
initial_frame_idx = 0
object_id = 1
input_points = np.array([[257, 176], [235, 286]])
input_labels = np.array([1, 0])
input_box = None

def run_sync(predictor,initial_frame_idx,object_id,input_points,input_labels,input_box,save_folder="data/test_video"):
    # run predict
    inference_state = predictor.init_state(video_path)
    frame_idx, _, masks_out = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=initial_frame_idx,
        obj_id=object_id,
        points=input_points,
        labels=input_labels,
        box=input_box,
    )
    initial_mask = masks_out[0:1, ...].cpu().numpy()
    video_masks = {frame_idx: initial_mask}
    propagation_generator = predictor.propagate_in_video(inference_state)
    for f_idx, _, m_out in propagation_generator:
        video_masks[f_idx] = (
            torch.where(torch.squeeze(m_out[0:1, ...]) > 0, 1, 0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    # write result
    frame_idx = 0
    cap = cv2.VideoCapture(video_path)
    writer = gen_video_writer(
        save_folder,
        frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            writer(None)
            break
        save_masks(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            [video_masks.get(frame_idx, None)],
            deal_func=writer,
        )
        frame_idx += 1
    cap.release()

t1=threading.Thread(target=run_sync,args=(predictor,initial_frame_idx,object_id,input_points,input_labels,input_box,"data/test_video1"))
t2=threading.Thread(target=run_sync,args=(predictor,initial_frame_idx,object_id,input_points,input_labels,input_box,"data/test_video2"))
t1.start()
t2.start()
t1.join()
t2.join()
