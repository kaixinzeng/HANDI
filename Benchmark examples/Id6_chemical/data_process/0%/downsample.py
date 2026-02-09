#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
downsample.py
Downsample time series data (supports npy and csv formats)

Features:
- Input files can be .npy or .csv
- Outputs downsampled .npy / .csv
- Auto-detects data dimensions: (N,T,D) or (T,D)
- Customizable stride (how many steps to skip per sample point)

Usage examples:
    python downsample.py --input data.npy --output data_down.npy --stride 5
    python downsample.py --input data.csv --output data_down.csv --stride 10
"""

import numpy as np
import argparse
import os
import pandas as pd

def downsample_array(data, stride):
    """Downsample numpy array"""
    if data.ndim == 3:
        data_down = data[:, ::stride, :]
    elif data.ndim == 2:
        data_down = data[::stride, :]
    else:
        raise ValueError(f"Unsupported data dimension: {data.shape}")
    return data_down

def downsample_file(input_path, output_path, stride):
    """Downsample based on file type"""
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".npy":
        data = np.load(input_path)
        data_down = downsample_array(data, stride)
        np.save(output_path, data_down)
        print(f"Downsampling complete: {input_path} -> {output_path}")
        print(f"Original shape: {data.shape}, Downsampled shape: {data_down.shape}")

    elif ext == ".csv":
        df = pd.read_csv(input_path)
        df_down = df.iloc[::stride, :].reset_index(drop=True)
        df_down.to_csv(output_path, index=False)
        print(f"Downsampling complete: {input_path} -> {output_path}")
        print(f"Original rows: {len(df)}, Downsampled rows: {len(df_down)}")

    else:
        raise ValueError("Only .npy or .csv files are supported")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsampling tool")
    parser.add_argument("--input", type=str, default="id6.npy", help="Input file path (.npy or .csv)")
    parser.add_argument("--output", type=str, default="id6_downsample_45.npy", help="Output file path")
    parser.add_argument("--stride", type=int, default=45, help="Downsampling stride")
    args = parser.parse_args()

    downsample_file(args.input, args.output, args.stride)
