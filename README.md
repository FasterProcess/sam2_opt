# sam2_opt
optimize sam2 with tensorrt


# Download models

```bash
cd sam2/checkpoints

chmod +x download_opt.sh
./download_opt.sh
./download_ckpts.sh # download raw models with no change
```

# usage

## image predictor

```python
import sys
sys.path.insert(0, "sam2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# only support large version
sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)  # type:SAM2ImagePredictor

predictor.speedup()     # After initializing the predictor, simply insert one line of code afterwards.

# use predictor like raw version

# predictor.speedup("torch")        # reset to raw version, which support other model version, such as tiny
```

## Video predictor

```python
import sys
sys.path.insert(0, "sam2")
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

# only support large version
sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(
    model_cfg, sam2_checkpoint, device=device
)  # type:SAM2VideoPredictor

predictor.speedup()     # After initializing the predictor, simply insert one line of code afterwards.

# use predictor like raw version

# predictor.speedup("torch")        # reset to raw version, which support other model version, such as tiny
```

# how to compatible with sam2 installed by pip

```python
# add below code to your first run-line
import sys
sys.path.insert(0, "sam2")
```