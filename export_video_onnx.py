import sys
import os
import torch
import onnx

# Add the sam2 directory to the Python path
sys.path.insert(0, "sam2")

from sam2.build_sam import build_sam2_video_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
onnx_path = "models"

# Model configuration
sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# --- Helper Function ---

def simplify_and_save(onnx_model, save_path):
    """Helper function to simplify and save the ONNX model."""
    try:
        from onnxsim import simplify
        print(f"Simplifying ONNX model: {save_path}")
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, save_path)
        print(f"Successfully simplified and saved to {save_path}")
    except Exception as e:
        print(f"Error during simplification: {e}. Saving the original model.")
        onnx.save(onnx_model, save_path)

# --- Export Functions with Assertions ---

@torch.no_grad()
def export_prompt_encoder(onnx_name="video_prompt_encoder.onnx", simplify_onnx=True, override=False):
    """
    Exports the Prompt Encoder and verifies its response to a dynamic number of points.
    """
    global predictor, onnx_path
    print("\n--- [Prompt Encoder] Assertion and Export ---")
    os.makedirs(onnx_path, exist_ok=True)
    save_path = os.path.join(onnx_path, onnx_name)
    if os.path.exists(save_path) and not override:
        print(f"Skipping, {save_path} already exists.")
        return

    prompt_encoder = predictor.sam_prompt_encoder

    # --- Assertion Phase ---
    print(">>> 1. Assertion Phase: Verifying dynamic number of points...")

    # Scenario 1: Baseline input with 2 points
    print("  - Running with baseline input (N=1, 2 points)...")
    points_coords_base = torch.randint(0, 1024, (1, 2, 2), dtype=torch.float, device=device)
    points_labels_base = torch.tensor([[1, 0]], dtype=torch.int, device=device)
    args_base = ((points_coords_base, points_labels_base), None, None)
    base_sparse_emb, base_dense_emb = prompt_encoder(*args_base)
    print(f"    Base output shapes: sparse_emb={base_sparse_emb.shape}, dense_emb={base_dense_emb.shape}")

    # Scenario 2: Variant input with 5 points
    print("  - Running with variant input (N=1, 5 points)...")
    points_coords_variant = torch.randint(0, 1024, (1, 5, 2), dtype=torch.float, device=device)
    points_labels_variant = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.int, device=device)
    args_variant = ((points_coords_variant, points_labels_variant), None, None)
    variant_sparse_emb, variant_dense_emb = prompt_encoder(*args_variant)
    print(f"    Variant output shapes: sparse_emb={variant_sparse_emb.shape}, dense_emb={variant_dense_emb.shape}")
    
    # Assertions: Sparse embeddings should change with num_points, dense should not.
    assert base_sparse_emb.shape[1] != variant_sparse_emb.shape[1], "Sparse embeddings token dimension should be dynamic with num_points."
    assert base_dense_emb.shape == variant_dense_emb.shape, "Dense embeddings shape should be static regardless of num_points."
    print("  - Assertions PASSED: Model correctly handles a dynamic number of points.")

    # --- Export Phase ---
    print("\n>>> 2. Export Phase...")
    # Using the variant args for export to ensure the dynamic axis is captured.
    torch.onnx.export(prompt_encoder, args_variant, save_path, export_params=True, opset_version=17, do_constant_folding=True,
                      input_names=["point_coords", "point_labels"],
                      output_names=["sparse_embeddings", "dense_embeddings"],
                      dynamic_axes={"point_coords": {0: "N", 1: "num_points"},
                                    "point_labels": {0: "N", 1: "num_points"}, 
                                    "sparse_embeddings": {0: "N", 1: "num_tokens"}, # Making token dimension dynamic
                                    "dense_embeddings": {0: "N"}})

    print(f"Exported to {save_path}")
    if simplify_onnx:
        simplify_and_save(onnx.load(save_path), save_path.replace(".onnx", "_opt.onnx"))


@torch.no_grad()
def export_memory_encoder(onnx_name="video_memory_encoder.onnx", simplify_onnx=True, override=False):
    """
    Exports the Memory Encoder and verifies its response to dynamic batch sizes.
    This exports the `inference_memory` method by temporarily modifying the forward pass.
    """
    global predictor, onnx_path
    print("\n--- [Memory Encoder] Assertion and Export ---")
    os.makedirs(onnx_path, exist_ok=True)
    save_path = os.path.join(onnx_path, onnx_name)
    if os.path.exists(save_path) and not override:
        print(f"Skipping, {save_path} already exists.")
        return

    module = predictor.memory_encoder
    original_forward = module.forward
    
    # Temporarily set the forward method to the one we want to export
    module.forward = module.inference_memory

    try:
        # --- Assertion Phase ---
        print(">>> 1. Assertion Phase: Verifying dynamic batch size...")

        # Scenario 1: Baseline input with batch size N=1
        print("  - Running with baseline input (N=1)...")
        pix_feat_base = torch.randn(1, 256, 64, 64, device=device)
        mask_base = torch.rand(1, 1, 1024, 1024, device=device)
        args_base = (pix_feat_base, mask_base)
        base_outputs = module.forward(*args_base)
        base_shapes = [o.shape for o in base_outputs]
        print(f"    Baseline output shapes: {base_shapes}")

        # Scenario 2: Variant input with batch size N=3
        print("  - Running with variant input (N=3)...")
        pix_feat_variant = torch.randn(3, 256, 64, 64, device=device)
        mask_variant = torch.rand(3, 1, 1024, 1024, device=device)
        args_variant = (pix_feat_variant, mask_variant)
        variant_outputs = module.forward(*args_variant)
        variant_shapes = [o.shape for o in variant_outputs]
        print(f"    Variant output shapes: {variant_shapes}")

        # Assertions
        print("  - Asserting shape changes...")
        for i, (base_shape, var_shape) in enumerate(zip(base_shapes, variant_shapes)):
            assert var_shape[0] == 3, f"Output {i} batch dim should be 3, but got {var_shape[0]}"
            assert base_shape[1:] == var_shape[1:], f"Output {i} non-batch dims should be static. Base: {base_shape[1:]}, Variant: {var_shape[1:]}"
        print("  - Assertions PASSED: Model correctly handles dynamic batch size.")

        # --- Export Phase ---
        print("\n>>> 2. Export Phase...")
        torch.onnx.export(
            module,
            args_variant, # Use the variant args to ensure the dynamic axis is captured
            save_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["pixel_features", "mask_for_memory"],
            output_names=["mask_memory_features", "mask_memory_pos_enc"],
            dynamic_axes={
                "pixel_features": {0: "N"},
                "mask_for_memory": {0: "N"},
                "mask_memory_features": {0: "N"},
                "mask_memory_pos_enc": {0: "N"}
            }
        )
        print(f"Exported to {save_path}")
        if simplify_onnx:
            simplify_and_save(onnx.load(save_path), save_path.replace(".onnx", "_opt.onnx"))

    finally:
        # ALWAYS restore the original forward method to avoid side-effects
        module.forward = original_forward
        print("  - Restored original forward method for MemoryEncoder.")


if __name__ == "__main__":
    print("Building and loading SAM2VideoPredictor model...")
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    predictor.eval()
    print("Model loaded successfully.")

    export_functions = [
        export_prompt_encoder,
        export_memory_encoder,
    ]

    for export_func in export_functions:
        try:
            # Running with override=True for demonstration
            export_func(override=True, simplify_onnx=True) 
        except Exception as e:
            import traceback

            print(f"\n[FATAL ERROR] Failed to run {export_func.__name__}: {e}")
            traceback.print_exc()
            print("Skipping this module and continuing with the next one.")

    print("\n--- ONNX Rigorous Export Process Finished ---")
    print(f"All attempted models have been exported to the '{onnx_path}' directory.")