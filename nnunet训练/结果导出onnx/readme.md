# 导出onnx模型

1. 配置global_dir的两个参数：save_dir（原始参数保存目录）；onnx_dir（转化onnx保存目录）

2. 执行config_download.py：将所有非商业task相关模型配置下载到save_dir
3. 执行transform_to_onnx.py：save_dir中配置导出转换为onnx并复制部分配置信息到onnx_dir
4. 手动删除save_dir（可选）