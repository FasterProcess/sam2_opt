# video.py

import sys
import time

from ytools.bench import test_torch_cuda_time

# Add the sam2 directory to the Python path to import its modules successfully
sys.path.insert(0, "sam2")

import os
import shutil
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor


def draw_mask(img, mask, color=[0, 0, 255], alpha=0.6):
    squeezed_mask = np.squeeze(mask)
    if squeezed_mask.ndim != 2:
        raise ValueError(
            f"Cannot convert input mask to a 2D (H, W) shape. Original shape: {mask.shape}, after squeeze: {squeezed_mask.shape}"
        )
    bool_mask = squeezed_mask > 0
    out_img = img.copy()
    color_layer = np.zeros_like(out_img, dtype=np.uint8)
    color_layer[bool_mask] = color
    out_img = cv2.addWeighted(out_img, 1, color_layer, alpha, 0)
    return out_img


def save_video_masks(video_path, masks_dict, output_fold="data/test_video"):
    if os.path.exists(output_fold):
        shutil.rmtree(output_fold)
    os.makedirs(output_fold, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Saving video frames")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in masks_dict:
            mask = masks_dict[frame_idx]
            masked_frame = draw_mask(frame.copy(), mask)
            bw_mask = (np.squeeze(mask) > 0).astype(np.uint8)
            mask_save_path = os.path.join(output_fold, f"{frame_idx:05d}_mask.png")
            cv2.imwrite(mask_save_path, bw_mask * 255)
            gen_save_path = os.path.join(output_fold, f"{frame_idx:05d}_gen.png")
            cv2.imwrite(gen_save_path, masked_frame)
        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()


@test_torch_cuda_time()
def run_segmentation(
    predictor: SAM2VideoPredictor,
    video_path,
    frame_idx,
    obj_id,
    points=None,
    labels=None,
    box=None,
):
    print("Step 1: Initializing inference state...")
    inference_state = predictor.init_state(video_path)
    print(f"Step 2: Adding initial prompt at frame {frame_idx}...")
    frame_idx_out, _, masks_out = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
        box=box,
    )
    initial_mask = masks_out[0:1, ...].cpu().numpy()
    all_masks = {frame_idx_out: initial_mask}
    print("Initial mask obtained.")
    print("Step 3: Propagating masks...")
    propagation_generator = predictor.propagate_in_video(inference_state)
    for f_idx, _, m_out in propagation_generator:
        all_masks[f_idx] = m_out[0:1, ...].cpu().numpy()
    print("Mask propagation complete.")
    return all_masks


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_tiny.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    predictor = build_sam2_video_predictor(
        model_cfg, sam2_checkpoint, device=device
    )  # type:SAM2VideoPredictor

    predictor.speedup("trt")

    # Option B: ONNX Runtime backend
    # print("\n--- Using ONNX Runtime backend ---")
    # # Configure onnx paths
    # onnx_paths ={
    #     "video_image_encoder": "models/forward_image_opt.onnx",
    #     "video_prompt_encoder": "models/video_prompt_encoder_opt.onnx",
    #     "video_mask_decoder": "models/image_mask_decoder_opt.onnx",
    #     "video_memory_encoder": "models/video_memory_encoder_opt.onnx",
    #     "video_memory_attention": "models/memory_attention_opt.onnx", # <-- new
    # }
    # # --- new ---
    # print("\n--- Set video_memory_attention backend ---")
    # # memory_attention 是 SAM2Base 的一个属性，可以通过 predictor 访问
    # predictor.memory_attention.set_runtime_backend(
    #     backend="onnxruntime",
    #     args={
    #         "model_paths": [onnx_paths["video_memory_attention"]],
    #         "providers": [
    #             "TensorrtExecutionProvider",
    #             "CUDAExecutionProvider",
    #             "CPUExecutionProvider",
    #         ],
    #     },
    # )

    # print("\n--- Set video_image_encoder backend ---")
    # predictor.set_runtime_backend(
    #     backend="onnxruntime",
    #     args={
    #         "model_paths": [onnx_paths["video_image_encoder"]],
    #         "providers": [
    #             "TensorrtExecutionProvider",
    #             "CUDAExecutionProvider",
    #             "CPUExecutionProvider",
    #         ],
    #     },
    # )
    # print("\n--- Set video_mask_decoder backend ---")
    # predictor.sam_mask_decoder.set_runtime_backend(
    #     backend="onnxruntime",
    #     args={
    #         "model_paths": [onnx_paths["video_mask_decoder"]],
    #         "providers": [
    #             "TensorrtExecutionProvider",
    #             "CUDAExecutionProvider",
    #             "CPUExecutionProvider",
    #         ],
    #     },
    # )
    # print("\n--- Set video_prompt_encoder backend ---")
    # predictor.sam_prompt_encoder.set_runtime_backend(
    #     backend="onnxruntime",
    #     args={
    #         "model_paths": [onnx_paths["video_prompt_encoder"]],
    #         "providers": [
    #             "TensorrtExecutionProvider",
    #             "CUDAExecutionProvider",
    #             "CPUExecutionProvider",
    #         ],
    #     },
    # )

    # print("\n--- Set video_memory_encoder backend ---")
    # predictor.memory_encoder.set_runtime_backend(
    #     backend="onnxruntime",
    #     args={
    #         "model_paths": [onnx_paths["video_memory_encoder"]],
    #         "providers": [
    #             "TensorrtExecutionProvider",
    #             "CUDAExecutionProvider",
    #             "CPUExecutionProvider",
    #         ],
    #     },
    # )
    # print("Successfully set onnxruntime_backend")

    # --- 3. Run Inference ---
    video_path = "./sam2/notebooks/videos/bedroom.mp4"
    initial_frame_idx = 0
    object_id = 1

    # print("\nTest Mode: Bounding Box + Foreground Point")
    # input_points = np.array([[257, 176]])
    # input_labels = np.array([1])
    # input_box = np.array([161, 138, 291, 415])

    # final_masks = run_segmentation(
    #     predictor, video_path, initial_frame_idx, object_id,
    #     points=input_points, labels=input_labels, box=input_box
    # )
    # --- 1: Multi-point input (foreground + background) ---
    print("Test Mode: Multi-point input")
    input_points = np.array(
        [[257, 176], [235, 286]]
    )  # Foreground point + Background point
    input_labels = np.array([1, 0])  # Foreground label + Background label
    final_masks = run_segmentation(
        predictor,
        video_path,
        initial_frame_idx,
        object_id,
        points=input_points,
        labels=input_labels,
    )

    # --- 2: Bounding box input ---
    # print("Test Mode: Bounding box input")
    # input_box = np.array([161, 138, 291, 415]) # [x1, y1, x2, y2]
    # final_masks = run_segmentation(
    #     predictor, video_path, initial_frame_idx, object_id,
    #     box=input_box
    # )

    # --- 3: Bounding box + one foreground point ---
    # print("Test Mode: Bounding box + Foreground point")
    # input_points = np.array([[257, 176]]) # Point on the child
    # input_labels = np.array([1])
    # input_box = np.array([161, 138, 291, 415]) # Box roughly enclosing the child
    # final_masks = run_segmentation(
    #     predictor, video_path, initial_frame_idx, object_id,
    #     points=input_points, labels=input_labels, box=input_box
    # )

    # --- 4. Save Results and Timing ---
    save_video_masks(video_path, final_masks, output_fold="data/test_video")
