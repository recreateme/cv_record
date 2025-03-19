import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from totalsegmentator.python_api import totalsegmentator
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_onnx_model(onnx_file_path):
    with open(onnx_file_path, 'rb') as f:
        onnx_model = f.read()
    return onnx_model


def build_engine(onnx_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 加载 ONNX 模型
    onnx_model = load_onnx_model(onnx_file_path)
    if not parser.parse(onnx_model):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

    # 配置构建选项
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 30  # 1GB

    # 构建引擎
    engine = builder.build_cuda_engine(network)
    return engine


def infer(engine, input_data):
    # 创建执行上下文
    context = engine.create_execution_context()

    # 分配输入输出内存
    input_shape = (1,) + engine.get_binding_shape(0)[1:]  # 例如 (1, 3, 224, 224)
    output_shape = engine.get_binding_shape(1)

    input_size = trt.volume(input_shape) * engine.max_batch_size
    output_size = trt.volume(output_shape) * engine.max_batch_size

    d_input = cuda.mem_alloc(input_size * np.float32().itemsize)
    d_output = cuda.mem_alloc(output_size * np.float32().itemsize)

    # 将输入数据复制到设备
    cuda.memcpy_htod(d_input, input_data)

    # 执行推理
    context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])

    # 从设备复制输出数据
    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output_data, d_output)

    return output_data


if __name__ == "__main__":
    onnx_file_path = "model.onnx"  # 替换为你的 ONNX 模型路径
    engine = build_engine(onnx_file_path)

    # 准备输入数据
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)  # 根据模型输入形状调整

    output_data = infer(engine, input_data)
    print("Output:", output_data)
