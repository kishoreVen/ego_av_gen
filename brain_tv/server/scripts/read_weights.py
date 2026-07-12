#!/usr/bin/env python3
"""Read safetensors weights and output JSON for the watch UI."""

import argparse
import json
import sys

import numpy as np
from safetensors.numpy import load_file


def get_metadata(filepath: str) -> list:
    """Return metadata for all tensors in the file."""
    tensors = load_file(filepath)
    result = []
    for name, arr in tensors.items():
        result.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "num_elements": int(arr.size),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
        )
    return result


def get_tensor_heatmap(filepath: str, tensor_name: str, max_size: int = 128) -> dict:
    """Return a downsampled 2D representation of a tensor for heatmap rendering."""
    tensors = load_file(filepath)
    if tensor_name not in tensors:
        return {"error": f"Tensor '{tensor_name}' not found"}

    arr = tensors[tensor_name].astype(np.float32)
    original_shape = list(arr.shape)

    # Flatten to 2D: (first_dim, product_of_rest)
    if arr.ndim == 0:
        data_2d = arr.reshape(1, 1)
    elif arr.ndim == 1:
        data_2d = arr.reshape(1, -1)
    else:
        data_2d = arr.reshape(arr.shape[0], -1)

    # Downsample to max_size x max_size via uniform index sampling
    h, w = data_2d.shape
    if h > max_size:
        indices = np.linspace(0, h - 1, max_size, dtype=int)
        data_2d = data_2d[indices]
    if w > max_size:
        indices = np.linspace(0, w - 1, max_size, dtype=int)
        data_2d = data_2d[:, indices]

    return {
        "name": tensor_name,
        "original_shape": original_shape,
        "heatmap_shape": list(data_2d.shape),
        "min": float(np.min(data_2d)),
        "max": float(np.max(data_2d)),
        "mean": float(np.mean(data_2d)),
        "std": float(np.std(data_2d)),
        "values": data_2d.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("--mode", choices=["metadata", "tensor"], default="metadata")
    parser.add_argument("--name", default=None, help="Tensor name for tensor mode")
    parser.add_argument("--max-size", type=int, default=128)
    args = parser.parse_args()

    if args.mode == "metadata":
        result = get_metadata(args.filepath)
    elif args.mode == "tensor":
        if not args.name:
            print(json.dumps({"error": "--name required for tensor mode"}))
            sys.exit(1)
        result = get_tensor_heatmap(args.filepath, args.name, args.max_size)

    json.dump(result, sys.stdout)
