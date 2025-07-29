from ytools.tensorrt import save_engine_mixed_inputs
import os

os.makedirs("models/engine", exist_ok=True)


def transfer_set_image_e2e_engine():
    calibration_dataset_path = "datasets/image/mini_batch"

    input_model_path = "models/set_image_e2e_opt.onnx"
    output_model_path = "models/engine/set_image_e2e_opt_fp16.engine"

    print(f"start to transfer {input_model_path}")

    save_engine_mixed_inputs(
        input_model_path,
        output_model_path,
        max_workspace_size=15 * (1 << 30),
        fp16_mode=True,
        int8_mode=False,
        dynamic_axes={"image": {"min": {0: 1}, "opt": {0: 1}, "max": {0: 10}}},
        calibrator=None,
    )

    print(f"success transfer {input_model_path} to {output_model_path}")


def transfer_forward_image_engine():
    input_model_path = "models/forward_image_opt.onnx"
    output_model_path = "models/engine/forward_image_opt_fp16.engine"

    print(f"start to transfer {input_model_path}")

    if save_engine_mixed_inputs(
        input_model_path,
        output_model_path,
        max_workspace_size=15 * (1 << 30),
        fp16_mode=True,
        int8_mode=False,
        dynamic_axes={"image": {"min": {0: 1}, "opt": {0: 1}, "max": {0: 10}}},
        calibrator=None,
    ):
        print(f"success transfer {input_model_path} to {output_model_path}")


def transfer_memory_attention_engine():
    input_model_path = "models/memory_attention.onnx"
    output_model_path = "models/memory_attention_fp16.engine"

    print(f"start to transfer {input_model_path}")

    if save_engine_mixed_inputs(
        input_model_path,
        output_model_path,
        max_workspace_size=15 * (1 << 30),
        fp16_mode=True,
        int8_mode=False,
        dynamic_axes={
            "curr": {"min": {1: 1}, "opt": {1: 1}, "max": {1: 1}},
            "memory": {
                "min": {0: 4096, 1: 1},
                "opt": {0: 28736, 1: 1},
                "max": {0: 28736, 1: 1},
            },
            "curr_pos": {"min": {1: 1}, "opt": {1: 1}, "max": {1: 1}},
            "memory_pos": {
                "min": {0: 4096, 1: 1},
                "opt": {0: 28736, 1: 1},
                "max": {0: 28736, 1: 1},
            },
        },
        calibrator=None,
    ):
        print(f"success transfer {input_model_path} to {output_model_path}")


if __name__ == "__main__":
    # transfer_set_image_e2e_engine()
    # transfer_forward_image_engine()
    transfer_memory_attention_engine()
