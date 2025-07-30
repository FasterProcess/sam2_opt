# video.py (Modified to support switching between original and quantized models)

import sys
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import glob
from PIL import Image
from typing import Tuple, List
import random

# Add the sam2 directory to the Python path to make its modules importable.
sys.path.insert(0, "sam2")

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

def save_masks_for_evaluation(
    masks: List[torch.Tensor],
    output_dir_base: str,
    video_name: str,
    obj_id_str: str
) -> None:
    """
    Saves a list of predicted masks to disk for evaluation.

    Each mask is saved as a separate PNG image in a structured directory format:
    `output_dir_base/video_name/obj_id_str/frame_index.png`

    Args:
        masks (List[torch.Tensor]): A list of mask tensors, where each tensor
                                    corresponds to a frame.
        output_dir_base (str): The root directory where predictions will be saved.
        video_name (str): The name of the video, used for creating a subfolder.
        obj_id_str (str): The identifier for the tracked object, used for creating
                          a sub-subfolder.
    """
    # Construct the full path for the specific video and object ID.
    video_output_dir = os.path.join(output_dir_base, video_name, obj_id_str)
    os.makedirs(video_output_dir, exist_ok=True) # Create the directory if it doesn't exist.

    # Iterate through each mask and save it as an image.
    for frame_idx, mask_tensor in enumerate(masks):
        # Convert the tensor to a NumPy array on the CPU and remove single-dimensional entries.
        mask_numpy = mask_tensor.cpu().numpy().squeeze()
        # Threshold the mask (values > 0 become 1, others 0) and scale to 255 for saving as an image.
        # mask_image = (mask_numpy * 255).astype(np.uint8)
        mask_image = (mask_numpy > 0).astype(np.uint8) * 255
        # Convert the probability map (0.0-1.0) to a grayscale image (0-255).
        # Create a PIL image from the NumPy array in grayscale ('L') mode.
        pil_image = Image.fromarray(mask_image, 'L')
        # Define the full save path for the current frame's mask.
        save_path = os.path.join(video_output_dir, f"{frame_idx:05d}.png")
        # Save the image.
        pil_image.save(save_path)

def run_segmentation_with_gt_mask(
    predictor: SAM2VideoPredictor,
    video_path: str,
    first_frame_gt_mask: np.ndarray,
    obj_id: int
) -> List[torch.Tensor]:
    """
    Runs video object segmentation starting from a given ground truth mask on the first frame.

    This function initializes the predictor for a video, provides the first frame's mask
    to identify the object, and then propagates that mask through the rest of the video.

    Args:
        predictor (SAM2VideoPredictor): The SAM2 video predictor instance.
        video_path (str): The path to the input video file.
        first_frame_gt_mask (np.ndarray): The ground truth mask for the object in the first frame.
        obj_id (int): The unique integer ID for the object being tracked.

    Returns:
        List[torch.Tensor]: A list of all predicted mask tensors for the entire video.
    """
    with torch.inference_mode():
        # Initialize the predictor's state for the new video. This loads the video and prepares for tracking.
        inference_state = predictor.init_state(video_path)

        initial_frame_idx = 0
        # Add the first ground truth mask to the predictor to start the tracking process.
        _, _, pred_masks = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=initial_frame_idx,
            mask=first_frame_gt_mask,
            obj_id=obj_id
        )
        # This list will store all masks, starting with the one from the first frame.
        all_masks = [pred_masks]

        # Create a generator that will yield propagated masks for subsequent frames.
        propagation_generator = predictor.propagate_in_video(inference_state)

        for _, _, propagated_masks in propagation_generator:
                all_masks.append(propagated_masks)
            
    return all_masks

def get_first_frame_gt_mask(gt_object_path: str) -> Tuple[np.ndarray | None, int]:
    """
    Loads the ground truth mask for the first frame of a specific object.

    Args:
        gt_object_path (str): The directory path containing the ground truth mask images
                              for a single object, e.g., '.../video_name/01/'.

    Returns:
        Tuple[np.ndarray | None, int]: A tuple containing:
            - The binary NumPy array of the first frame's mask (or None if not found/error).
            - The total number of mask files found for this object.
    """
    # Find all PNG files in the directory and sort them to ensure the first frame is correct.
    mask_files = sorted(glob.glob(os.path.join(gt_object_path, '*.png')))
    if not mask_files:
        return None, 0

    first_mask_path = mask_files[0] # '/root/workspace/code/workspace/wuchenfan/sam2_opt/sav_val/Annotations_6fps/sav_000262/000/00000.png'
    try:
        # Open the first mask image and convert it to grayscale.
        mask_image = Image.open(first_mask_path).convert('L')
        # Convert the PIL image to a NumPy array.
        gt_mask = np.array(mask_image)
        # Binarize the mask: pixels with intensity > 128 become 1, others 0.
        gt_mask = (gt_mask > 128).astype(np.uint8)
        return gt_mask, len(mask_files)
    except Exception as e:
        print(f"Error loading GT mask {first_mask_path}: {e}")
        return None, 0

if __name__ == "__main__":
    # --- 1. Configuration ---
    USE_QUANTIZED_MODEL = True

    # --- Path Configuration ---
    # Define the base directory for all data and models.
    BASE_DATA_DIR = "/root/workspace/code/workspace/wuchenfan/sam2_opt"
    # Directory containing the ground truth annotation masks.
    GT_DIR = os.path.join(BASE_DATA_DIR, "sav_test/Annotations_6fps")
    # Directory containing the source video files.
    VIDEO_SRC_DIR = os.path.join(BASE_DATA_DIR, "sav_test/JPEGImages_24fps")

    # Select the output directory based on the model choice to prevent overwriting results.
    if USE_QUANTIZED_MODEL:
        OUTPUT_PRED_DIR = "output_predictions_trt"
    else:
        OUTPUT_PRED_DIR = "output_predictions_large_0"
        
    os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)
    print(f"--- Running in {'QUANTIZED' if USE_QUANTIZED_MODEL else 'ORIGINAL'} mode ---")
    print(f"--- Predictions will be saved to: {OUTPUT_PRED_DIR} ---")

    # --- 2. Model Preparation ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Unified model configuration for both original and quantized versions.
    # The base model structure and weights are loaded from these files.
    # large
    sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # tiny
    # sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_tiny.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    print("Building SAM2VideoPredictor model...")
    # This function constructs the model architecture from the config and loads the pretrained weights.
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device) # type:SAM2VideoPredictor

    # Set the backend based on the configuration switch.
    if USE_QUANTIZED_MODEL:
        print("\n--- Setting up Quantized TensorRT Backend ---")
        predictor.speedup("trt")

    else:
        print("\n--- Using Original PyTorch Backend ---")

    print("Model loaded and configured successfully.")
    
    # --- 3. Iterate and Process Videos ---
    print(f"\nScanning Ground Truth directory: '{GT_DIR}'")
    # Get a list of all subdirectories in the GT_DIR, each corresponding to a video.
    all_video_ids = sorted([d for d in os.listdir(GT_DIR) if os.path.isdir(os.path.join(GT_DIR, d))])
    print(f"Found {len(all_video_ids)} total videos with ground truth annotations.")

    # For testing purposes, limit the number of videos to process.
    # NUM_VIDEOS_TO_PROCESS = 2
    # if len(all_video_ids) > NUM_VIDEOS_TO_PROCESS:
    #     video_ids = all_video_ids[-NUM_VIDEOS_TO_PROCESS:]
    #     print(f"Selecting the first {len(video_ids)} videos to process for this run.")
    # else:
    #     video_ids = all_video_ids
    video_ids = all_video_ids

    for video_name in tqdm(video_ids, desc="Processing Videos"):
        # Path to the directory containing ground truth masks for the current video.
        gt_video_dir = os.path.join(GT_DIR, video_name) # '/root/workspace/code/workspace/wuchenfan/sam2_opt/sav_val/Annotations_6fps/sav_000262'
        # Path to the actual video file.
        video_path = os.path.join(VIDEO_SRC_DIR, video_name) # '/root/workspace/code/workspace/wuchenfan/sam2_opt/sav_val/JPEGImages_24fps/sav_000262'

        # Skip if the video file does not exist.
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found for '{video_name}', skipping.")
            continue
        
        # Each video can have multiple objects, each in its own subfolder (e.g., '000', '001').
        obj_id_folders = sorted([d for d in os.listdir(gt_video_dir) if os.path.isdir(os.path.join(gt_video_dir, d))])
        
        # Loop through each object in the current video.
        for obj_id_str in obj_id_folders:
            gt_object_path = os.path.join(gt_video_dir, obj_id_str)
            # Get the ground truth mask for the very first frame.
            first_frame_gt, _ = get_first_frame_gt_mask(gt_object_path)
            
            # If no mask is found, skip this object.
            if first_frame_gt is None:
                continue

            # Convert the object ID string to an integer.
            try:
                object_id_int = int(obj_id_str)
            except ValueError:
                continue

            tqdm.write(f"  -> Processing Video: {video_name}, Object ID: {obj_id_str}")

            all_masks = run_segmentation_with_gt_mask(
                predictor, 
                video_path,
                first_frame_gt, 
                object_id_int
            )
            
            if all_masks:
                save_masks_for_evaluation(
                    masks=all_masks,
                    output_dir_base=OUTPUT_PRED_DIR,
                    video_name=video_name,
                    obj_id_str=obj_id_str
                )
        
    print(f"\nAll {len(video_ids)} selected videos processed.")
    print(f"Predictions have been saved in '{OUTPUT_PRED_DIR}'.")