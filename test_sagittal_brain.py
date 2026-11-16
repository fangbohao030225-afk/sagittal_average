import numpy as np
import pytest
from sagittal_brain import run_averages

def test_simple_integer_average(tmp_path):
    """
    测试1：验证基础计算逻辑和维度。
    使用一个 3x2 的矩阵，我们能立刻口算出每一列的平均值。
    """
    # 定义 input 数组
    input_data = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    
    # 预期输出：因为 Charlene 用的是 axis=0（求每一列的平均值）
    # 第1列: (1+3+5)/3 = 3.0
    # 第2列: (2+4+6)/3 = 4.0
    # 并且输出格式被她转成了 shape (1, 2) 的行向量
    expected_output = np.array([[3.0, 4.0]])
    
    # 将输入数据写入临时 CSV 文件
    input_file = tmp_path / "input_int.csv"
    output_file = tmp_path / "output_int.csv"
    np.savetxt(input_file, input_data, delimiter=',', fmt='%d')
    
    # 运行 Charlene 的代码
    run_averages(file_input=str(input_file), file_output=str(output_file))
    
    # 读取她写出的结果
    result = np.loadtxt(output_file, delimiter=',', ndmin=2)
    
    # 检查形状和数值是否匹配
    assert result.shape == (1, 2), f"Expected shape (1, 2), but got {result.shape}"
    np.testing.assert_array_almost_equal(result, expected_output)


def test_with_float_data(tmp_path):
    """
    测试2：暴露她代码中的 `dtype=int` 缺陷。
    真实科学数据通常是浮点数，我们看看她的代码能不能处理。
    """
    input_data = np.array([
        [1.5, 2.5],
        [3.5, 4.5]
    ])
    
    input_file = tmp_path / "input_float.csv"
    np.savetxt(input_file, input_data, delimiter=',')
    
    # 因为她的代码写了 dtype=int，这里预期会抛出 ValueError（无法将 1.5 转为 int）
    with pytest.raises(ValueError):
        run_averages(file_input=str(input_file), file_output=str(tmp_path / "out.csv"))