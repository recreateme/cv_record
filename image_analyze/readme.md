#                                   胶片分析———安装部署使用说明

## 环境安装

**安装工具:**Anaconda 或 mini-Anaconda

**环境文件：**environment.yml

打开anaconda的prompt命令行，运行命令直接创建

```python
conda env create -f environment.yml
```

安装成功后生成**虚拟环境pt**，作为本项目的SDK，包含项目所有依赖

------

## 打包说明

**打包入口：**b.py文件

命令1（exe和依赖分离，速度快些）

```python
python -m PyInstaller -D b.py
```

命令2（打包成一个大的exe）

```
python -m PyInstaller -F b.py
```

------

## 使用说明

### 支持类型

| star    | 星形射野       |
| :------ | -------------- |
| fence   | 栅栏分析       |
| rec     | 方形           |
| gamma   | gamma分析      |
| overlap | 重叠的方形分析 |

### 使用方式

**输入：****exe文件同级目录**下存放待分析的**图片（类型名.png）**以及包含bounding box坐标的文件**analyrect.txt** （可选）

**输出：**生成文件**（类型名）**

命令行使用栅栏分析的例子如下，其他语言调用命令行实现

```
b.exe --type fence 
```

