# -*- coding: utf-8 -*-
"""
一个独立的脚本，用于读取 all_boxes.dat 文件，
按照 'simple' 摆的规则处理数据，
计算角度和角速度，
并将最终的 (theta, theta_dot) 数据同时保存为 .npy 和 .csv 文件 (CSV保留8位小数)。
"""

import pickle
import numpy as np
import math
import pysindy as ps
import pandas as pd

def process_and_save_pendulum_state():
    """
    主函数，执行数据处理和保存任务。
    """
    # --- 1. 定义 'simple' 摆的参数 ---
    try:
        all_boxes = pickle.load(open('all_boxes.dat', 'rb'))
        print(f"成功加载 'all_boxes.dat'，共 {all_boxes.shape[0]} 帧数据。")
    except FileNotFoundError:
        print("错误：'all_boxes.dat' 文件未找到。请确保它与此脚本在同一目录下。")
        return

    dt = 1/120
    mid_x = 227
    mid_y = 20
    first_frame = 140
    end_frame_offset = -130  # 使用您原始SINDy分析中的截取范围

    # --- 2. 按照您的规则计算角度 (theta) ---
    theta = []
    for i in range(all_boxes.shape[0]):
        x_o, y_o, w, h = all_boxes[i, :]
        x = x_o + w / 2 - mid_x
        y = y_o + h / 2 - mid_y
        theta.append(math.atan(x / y))
    
    # 转换为 NumPy 数组
    theta = np.asarray(theta)
    time = np.arange(0, all_boxes.shape[0] * dt, dt)

    # --- 3. 截取感兴趣的轨迹段 ---
    theta_segment = theta[first_frame:end_frame_offset].copy()
    time_segment = time[first_frame:end_frame_offset].copy()

    print(f"截取从第 {first_frame} 帧到倒数第 {-end_frame_offset} 帧的数据。")

    # --- 4. 使用 PySINDy 计算一阶导数 (角速度) ---
    theta_dot = ps.FiniteDifference(order=2, d=1)._differentiate(theta_segment, t=time_segment)
    
    # --- 5. 对齐数据 ---
    # 一阶导数(d=1)配合二阶精度(order=2)的中心差分，会在数组两端各去掉1个点。
    # 你的代码中 offset = 2，我们保持一致。
    offset = 2
    
    theta_final = theta_segment[offset:-offset]
    theta_dot_final = theta_dot[offset:-offset]

    # --- 6. 组合数据并同时保存为 .npy 和 .csv 文件 ---
    # 将两个一维数组沿列方向堆叠，形成 (n, 2) 的数组
    final_data_npy = np.stack([theta_final, theta_dot_final], axis=1)

    # --- 保存为 .npy 文件 ---
    npy_filename = 'pendulum_sindy_data.npy'
    np.save(npy_filename, final_data_npy)
    print(f"\n数据已成功处理并保存至 '{npy_filename}' (形状: {final_data_npy.shape})")

    # --- 保存为 .csv 文件 ---
    # 创建一个 pandas DataFrame 并指定列名
    df = pd.DataFrame(final_data_npy, columns=['theta', 'theta_dot'])
    
    csv_filename = 'pendulum_angle_velocity.csv'
    
    # 使用 float_format 参数指定小数位数, '%.8f' 表示保留8位小数
    df.to_csv(csv_filename, index=False, float_format='%.8f',header=False)
    
    print(f"数据也已保存至 '{csv_filename}' (形状: {df.shape})，并保留了8位小数。")

if __name__ == "__main__":
    process_and_save_pendulum_state()