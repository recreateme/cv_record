import onnxruntime as ort
import numpy as np

# 指定ONNX模型路径
model_path = r"D:\rs\onnx\task_291.onnx"

# 尝试创建一个使用GPU的会话
try:
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    print("Using GPU for inference")
except (RuntimeError, ValueError):
    # 如果GPU不可用或发生其他错误，则回退到CPU
    session = ort.InferenceSession(model_path)
    print("Using CPU for inference")

# 准备输入数据（这里假设输入数据已经准备好，并且是 NumPy ndarray 格式）
input_name = session.get_inputs()[0].name
input_data = np.random.random(size=(1, 1, 128, 128,128)).astype('float32')

# 运行推理
outputs = session.run(None, {input_name: input_data})

# 输出结果
print(outputs)
