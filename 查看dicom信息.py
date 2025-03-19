import pydicom
from pydicom.errors import InvalidDicomError

def view_rtstruct_contours(rtstruct_path):
    """
    查看RTSTRUCT文件中的轮廓信息。

    参数:
    rtstruct_path: RTSTRUCT DICOM文件的路径。
    """
    try:
        # 加载RTSTRUCT DICOM文件
        rtstruct = pydicom.dcmread(rtstruct_path)

        # 确保这是一个RT Structure Set
        if rtstruct.Modality != 'RTSTRUCT':
            raise ValueError("The provided file is not an RTSTRUCT DICOM file.")

        # 遍历结构集定义序列
        for structure in rtstruct.StructureSetROISequence:
            print(f"ROI Name: {structure.ROIName}")
            print(f"ROI Number: {structure.ROINumber}")
            print(f"ROI Display Color: {structure.ROIDisplayColor}")

            # 获取与当前ROI对应的轮廓序列
            contours = [contour for contour in rtstruct.ROIContourSequence if contour.ReferencedROINumber == structure.ROINumber]

            for contour in contours:
                print(f"  Contour Number: {contour.ContourNumber}")
                print(f"  Contour Geometric Type: {contour.ContourGeometricType}")
                print(f"  Number of Contour Points: {len(contour.ContourData) // 3}")
                # 打印轮廓点（可选）
                # 注意：每个轮廓点由三个值组成（x, y, z）
                # print(f"  Contour Points: {contour.ContourData}")

    except FileNotFoundError:
        print(f"Error: The file {rtstruct_path} does not exist.")
    except InvalidDicomError:
        print(f"Error: The file {rtstruct_path} is not a valid DICOM file.")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # 使用示例：
    rtstruct_file_path = r'D:\rs\Head\WXL01\CT.1.2.840.113619.2.278.3.279739146.378.1717632179.738.1.dcm'
    view_rtstruct_contours(rtstruct_file_path)
