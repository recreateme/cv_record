解决totalsegmentator无法打包为可执行文件的问题，
依赖参数和训练的模型文件等，使用onnx推导并导出所关注器官的分割结果
input:     list[organ_name]    dicom文件目录
output:     汇总的分割结果      rtstructure.dcm
