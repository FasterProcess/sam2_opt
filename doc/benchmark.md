# Image
## A100
  
| 推理方式             | 模型类型 | 耗时（s） | 加速比 | 是否量化 |
| -------------------- | -------- | --------- | ------ | -------- |
| torch                | tiny     | 0.050     | -      | 否       |
| torch                | Large    | 0.149     | 1.0    | 否       |
| onnxruntime_cuda     | Large    | 0.113     | 1.318  | 否       |
| onnxruntime_trt      | Large    | 0.080     | 1.863  | 否       |
| onnxruntime_e2e_cuda | Large    | 0.078     | 1.91   | 否       |
| onnxruntime_e2e_trt  | Large    | 0.061     | 2.442  | 否       |


## A6000
  
| 推理方式             | 模型类型 | 耗时（s） | 加速比 | 是否量化 |
| -------------------- | -------- | --------- | ------ | -------- |
| torch                | tiny     | 0.096     | -      | 否       |
| torch                | Large    | 0.256s    | 1.0    | 否       |
| onnxruntime_cuda     | Large    | 0.180     | 1.362  | 否       |
| onnxruntime_trt      | Large    | 0.172     | 1.488  | 否       |
| onnxruntime_e2e_cuda | Large    | 0.137     | 1.869  | 否       |
| onnxruntime_e2e_trt  | Large    | 0.117     | 2.188  | 否       |

## RTX4090
| 推理方式             | 模型类型 | 耗时（s） | 加速比 | 是否量化 |
| -------------------- | -------- | --------- | ------ | -------- |
| torch                | tiny     | 0.029     | -      | 否       |
| torch                | Large    | 0.092     | 1.0    | 否       |
| onnxruntime_cuda     | Large    | 0.092     | 1.0    | 否       |
| onnxruntime_trt      | Large    | 0.079     | 1.164  | 否       |
| onnxruntime_e2e_cuda | Large    | 0.083     | 1.108  | 否       |
| onnxruntime_e2e_trt  | Large    | 0.065     | 1.415  | 否       |
| onnxruntime_e2e_trt  | Large    | 0.024     | 3.833  | int8     |

# Video
