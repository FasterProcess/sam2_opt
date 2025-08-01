speedup large version via onnxruntime or tensorrt

# Download models

```bash
./download_ckpts.sh # download raw models with no change

cd checkpoints
chmod +x download_opt.sh
./download_opt.sh
```

# usage

## image predictor

```python
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