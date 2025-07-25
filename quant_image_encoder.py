from calibrator.image_encode import ImageEncodeCalibratorDataset
from ytools.tensorrt import save_engine, MyEntropyCalibrator
import os

os.makedirs("models/engine", exist_ok=True)


def quant_set_image_e2e_engine():
    calibration_dataset_path = "datasets/image"

    input_model_path = "models/set_image_e2e_opt.onnx"
    output_model_path = "models/engine/set_image_e2e_opt.engine"

    print(f"start to quant {input_model_path}")

    data_set = ImageEncodeCalibratorDataset(
        calibration_dataset_path,
        input_shapes=[(-1, 3, 1024, 1024)],
        batch_size=1,
        skip_frame=1,
        dataset_limit=1000,
        do_norm=False,
    )
    calibrator = MyEntropyCalibrator(
        data_loader=data_set, cache_file="models/engine/set_image_e2e_opt.cache"
    )

    save_engine(
        input_model_path,
        output_model_path,
        max_workspace_size=5 * (1 << 30),
        fp16_mode=True,
        int8_mode=True,
        min_batch=1,
        optimize_batch=1,
        max_batch=10,
        calibrator=calibrator,
    )

    print(f"success quant {input_model_path} to {output_model_path}")


def quant_forward_image_engine():
    calibration_dataset_path = "datasets/image"

    input_model_path = "models/forward_image_opt.onnx"
    output_model_path = "models/engine/forward_image_opt.engine"

    print(f"start to quant {input_model_path}")

    data_set = ImageEncodeCalibratorDataset(
        calibration_dataset_path,
        input_shapes=[(-1, 3, 1024, 1024)],
        batch_size=1,
        skip_frame=1,
        dataset_limit=1000,
        do_norm=True,
    )
    calibrator = MyEntropyCalibrator(
        data_loader=data_set, cache_file="models/engine/forward_image_opt.cache"
    )

    save_engine(
        input_model_path,
        output_model_path,
        max_workspace_size=5 * (1 << 30),
        fp16_mode=True,
        int8_mode=True,
        min_batch=1,
        optimize_batch=1,
        max_batch=10,
        calibrator=calibrator,
    )

    print(f"success quant {input_model_path} to {output_model_path}")


if __name__ == "__main__":
    quant_set_image_e2e_engine()
    # quant_forward_image_engine()
