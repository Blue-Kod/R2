#!/usr/bin/env python3
"""Convert LAS2 ONNX model to RKNN format for Rockchip NPU.

Run on the Orange Pi (Linux ARM64):
    pip install rknn-toolkit2
    python models/convert_to_rknn.py

Output: models/las2_s_640x384.rknn
"""

import argparse
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Convert LAS2 ONNX to RKNN")
    parser.add_argument("--onnx", default="models/las2_s_640x384.onnx",
                        help="Input ONNX model path")
    parser.add_argument("--output", default="models/las2_s_640x384.rknn",
                        help="Output RKNN model path")
    parser.add_argument("--target", default="rk3588",
                        help="Target platform (default: rk3588)")
    parser.add_argument("--quantize", choices=["dynamic", "float16", "none"],
                        default="float16",
                        help="Quantization mode (default: float16)")
    parser.add_argument("--dataset", default=None,
                        help="Calibration dataset .txt for INT8 quantization (optional)")
    args = parser.parse_args()

    try:
        from rknn.api import RKNN
    except ImportError:
        print("ERROR: rknn-toolkit2 not installed.")
        print("Install it: pip install rknn-toolkit2")
        sys.exit(1)

    if not os.path.isfile(args.onnx):
        print(f"ERROR: ONNX model not found: {args.onnx}")
        sys.exit(1)

    print(f"Loading ONNX model: {args.onnx}")
    rknn = RKNN(verbose=False)

    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.target,
    )

    print(f"Building RKNN model (quantize={args.quantize})...")
    quantize_enabled = args.quantize != "none"
    dataset = args.dataset if args.dataset and os.path.isfile(args.dataset) else None

    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print("ERROR: Failed to load ONNX model")
        sys.exit(1)

    ret = rknn.build(
        do_quantization=False,
        dataset=dataset,
    )
    if ret != 0:
        print("ERROR: Failed to build RKNN model")
        sys.exit(1)

    print(f"Exporting RKNN model: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print("ERROR: Failed to export RKNN model")
        sys.exit(1)

    rknn.release()

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"DONE: {args.output} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
