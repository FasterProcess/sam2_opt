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
| tensorrt_e2e         | Large    | 0.030     | 4.966  | fp16     |

## A6000
  
| 推理方式             | 模型类型 | 耗时（s） | 加速比 | 是否量化 |
| -------------------- | -------- | --------- | ------ | -------- |
| torch                | tiny     | 0.096     | -      | 否       |
| torch                | Large    | 0.256s    | 1.0    | 否       |
| onnxruntime_cuda     | Large    | 0.180     | 1.362  | 否       |
| onnxruntime_trt      | Large    | 0.172     | 1.488  | 否       |
| onnxruntime_e2e_cuda | Large    | 0.137     | 1.869  | 否       |
| onnxruntime_e2e_trt  | Large    | 0.117     | 2.188  | 否       |
| tensorrt_e2e         | Large    | 0.060     | 4.266  | fp16     |
| tensorrt_e2e         | Large    | 0.056     | 4.571  | int8     |

## RTX4090
| 推理方式             | 模型类型 | 耗时（s） | 加速比 | 是否量化 |
| -------------------- | -------- | --------- | ------ | -------- |
| torch                | tiny     | 0.029     | -      | 否       |
| torch                | Large    | 0.092     | 1.0    | 否       |
| onnxruntime_cuda     | Large    | 0.092     | 1.0    | 否       |
| onnxruntime_trt      | Large    | 0.079     | 1.164  | 否       |
| onnxruntime_e2e_cuda | Large    | 0.083     | 1.108  | 否       |
| onnxruntime_e2e_trt  | Large    | 0.065     | 1.415  | 否       |
| tensorrt_e2e         | Large    | 0.025     | 3.68   | fp16     |
| tensorrt_e2e         | Large    | 0.024     | 3.833  | int8     |

# Video

| 推理方式    | 模型类型 | 耗时（s） | 加速比 | 是否量化 |
| ----------- | -------- | --------- | ------ | -------- |
| torch       | tiny     | 0.077     | -      | 否       |
| torch       | Large    | 0.181     | 1.0    | 否       |
| onnxruntime | Large    | 0.116     | 1.56   | 否       |
| tensorrt    | Large    | 0.044     | 4.114  | fp16     |


## A6000
  
| 推理方式    | 模型类型 | 耗时（s） | 加速比 | 是否量化 |
| ----------- | -------- | --------- | ------ | -------- |
| torch       | tiny     | 0.103     | -      | 否       |
| torch       | Large    | 0.227     | 1.0    | 否       |
| onnxruntime | Large    | 0.205     | 1.107  | 否       |
| tensorrt    | Large    | 0.065     | 3.492  | fp16     |

## RTX4090

| 推理方式    | 模型类型 | 耗时（s） | 加速比 | 是否量化 |
| ----------- | -------- | --------- | ------ | -------- |
| torch       | tiny     | 0.078     | -      | 否       |
| torch       | Large    | 0.142     | 1.0    | 否       |
| onnxruntime | Large    | 0.111     | 1.279  | 否       |
| tensorrt    | Large    | 0.028     | 5.071  | fp16     |

## SA-V test on RTX4090
| 推理方式    | 模型类型 | J&F | J | F |
| ----------- | -------- | --------- | ------ | -------- |
| torch       | tiny     | 69.9      | 64.4   | 75.5     |
| torch       | Large    | 72.6      | 66.8   | 78.4     |
| tensorrt    | Large    | 72.5      | 66.7   | 78.3     |