import sys

sys.path.insert(0, "sam2")
import numpy as np
import torch
import cv2
from draw import save_masks, gen_image_writer
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# init predictor
sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=torch.device("cuda"))
predictor = SAM2ImagePredictor(sam2_model)
predictor.speedup()  # speedup

# test inputs
image = cv2.imread("./sam2/notebooks/images/truck.jpg")
image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

input_point = np.array([[500, 375], [502, 375]])
input_label = np.array([1, 1])

# run predict
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]

# sort and then first one get better score, length==3
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

save_masks(image, masks[0:1], deal_func=gen_image_writer("data/test_image"))
