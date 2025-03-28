#!/usr/bin/env python3
"""
venv環境情報エクスポートスクリプト
既存のvenv環境の詳細情報を収集し、再現可能な形式で出力します。
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime

import pkg_resources


def get_system_info():
    """システム情報を取得"""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_path": sys.executable,
    }


def get_installed_packages():
    """インストール済みのPythonパッケージ情報を取得"""
    return sorted(
        [
            {"name": pkg.key, "version": pkg.version}
            for pkg in pkg_resources.working_set
        ],
        key=lambda x: x["name"],
    )


def get_cuda_info():
    """CUDA情報を取得（可能な場合）"""
    cuda_info = {"available": False}

    # nvidiaのCUDAバージョン確認（nvidia-smiが利用可能な場合）
    try:
        output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        for line in output.split("\n"):
            if "CUDA Version:" in line:
                cuda_version = line.split("CUDA Version:")[1].strip()
                cuda_info["nvidia_smi_cuda_version"] = cuda_version
                cuda_info["available"] = True
                break
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # PyTorchからCUDA情報を取得
    try:
        import torch

        cuda_info["pytorch_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            cuda_info["pytorch_cuda_version"] = torch.version.cuda
            cuda_info["pytorch_device_count"] = torch.cuda.device_count()
            cuda_info["pytorch_device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            cuda_info["available"] = True
    except ImportError:
        cuda_info["pytorch_available"] = False

    # TensorFlowからCUDA情報を取得
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        cuda_info["tensorflow_gpu_available"] = len(gpus) > 0
        if len(gpus) > 0:
            cuda_info["tensorflow_gpu_count"] = len(gpus)
            cuda_info["available"] = True
            # TensorFlowのバージョン
            cuda_info["tensorflow_version"] = tf.__version__
    except ImportError:
        cuda_info["tensorflow_gpu_available"] = False

    return cuda_info


def get_environment_variables():
    """関連する環境変数を取得"""
    relevant_vars = [
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDNN_PATH",
        "LD_LIBRARY_PATH",
    ]

    return {var: os.environ.get(var) for var in relevant_vars if var in os.environ}


def export_requirements():
    """requirements.txtの形式でパッケージリストを出力"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.SubprocessError:
        return "Error generating requirements.txt"


def main():
    """メイン実行関数"""
    print("venv環境情報収集中...\n")

    # 情報収集
    env_data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "environment_variables": get_environment_variables(),
        "cuda_info": get_cuda_info(),
        "installed_packages": get_installed_packages(),
    }

    # JSONとして出力
    with open("environment_info.json", "w", encoding="utf-8") as f:
        json.dump(env_data, f, indent=2, ensure_ascii=False)

    # requirements.txtの生成
    requirements = export_requirements()
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)

    print("環境情報のエクスポートが完了しました！")
    print("- environment_info.json: 詳細な環境情報")
    print("- requirements.txt: インストールされているパッケージのリスト")

    # CUDA情報のサマリーを表示
    if env_data["cuda_info"]["available"]:
        print("\nCUDA情報:")
        if "nvidia_smi_cuda_version" in env_data["cuda_info"]:
            print(
                f"- NVIDIA CUDA バージョン: {env_data['cuda_info']['nvidia_smi_cuda_version']}"
            )
        if (
            "pytorch_available" in env_data["cuda_info"]
            and env_data["cuda_info"]["pytorch_available"]
        ):
            print(
                f"- PyTorch CUDA バージョン: {env_data['cuda_info']['pytorch_cuda_version']}"
            )
        if (
            "tensorflow_gpu_available" in env_data["cuda_info"]
            and env_data["cuda_info"]["tensorflow_gpu_available"]
        ):
            print(
                f"- TensorFlow GPU デバイス数: {env_data['cuda_info']['tensorflow_gpu_count']}"
            )
    else:
        print("\nCUDAは利用できないか、インストールされていません。")


if __name__ == "__main__":
    main()
