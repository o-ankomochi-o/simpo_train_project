#!/usr/bin/env python3
"""
venv環境再構築スクリプト
環境情報ファイルを利用して新しいvenv環境を構築します。
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def create_venv(venv_path):
    """新しいvenv環境を作成"""
    print(f"新しいvenv環境を作成中: {venv_path}")
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"venv作成エラー: {e}")
        return False


def install_packages(venv_path, requirements_path):
    """パッケージのインストール"""
    # venvのpipを特定
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_path, "Scripts", "pip")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")

    print(f"パッケージをインストール中...(これには時間がかかる場合があります)")
    try:
        # pipをアップグレード
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)

        # requirementsからインストール
        subprocess.run([pip_path, "install", "-r", requirements_path], check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"パッケージインストールエラー: {e}")
        return False


def check_cuda_compatibility(env_info, verbose=False):
    """CUDA互換性のチェック"""
    if not env_info.get("cuda_info", {}).get("available", False):
        print("元の環境ではCUDAが使用されていないため、互換性チェックをスキップします")
        return

    print("\nCUDA互換性チェック:")

    # システムでのCUDA利用可能性のチェック
    try:
        output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        print("✓ NVIDIA GPUが利用可能です")

        original_cuda = env_info.get("cuda_info", {}).get(
            "nvidia_smi_cuda_version", "不明"
        )
        if "CUDA Version:" in output:
            current_cuda = output.split("CUDA Version:")[1].strip().split()[0]
            print(f"- 元の環境のCUDAバージョン: {original_cuda}")
            print(f"- 現在の環境のCUDAバージョン: {current_cuda}")

            if verbose:
                print("\nnvidia-smi 出力:")
                print(output)
        else:
            print("警告: 現在のCUDAバージョンを検出できませんでした")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("✗ NVIDIA GPUまたはnvidia-smiが見つかりません")
        print(
            "  CUDA対応が必要な場合は、適切なNVIDIAドライバとCUDAをインストールしてください"
        )


def main():
    parser = argparse.ArgumentParser(description="venv環境を再構築するツール")
    parser.add_argument(
        "--env-info", default="environment_info.json", help="環境情報JSONファイルパス"
    )
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="requirements.txtファイルパス",
    )
    parser.add_argument(
        "--venv-path", default="new_venv", help="作成するvenv環境のパス"
    )
    parser.add_argument("--verbose", action="store_true", help="詳細情報を表示")
    args = parser.parse_args()

    # 環境情報を読み込み
    try:
        with open(args.env_info, "r", encoding="utf-8") as f:
            env_info = json.load(f)

        # requirementsファイルの存在チェック
        if not os.path.exists(args.requirements):
            print(f"エラー: {args.requirements} が見つかりません")
            return 1

        print("元の環境情報:")
        print(f"- Python: {env_info['system_info']['python_version']}")
        print(f"- プラットフォーム: {env_info['system_info']['platform']}")
        print(f"- パッケージ数: {len(env_info['installed_packages'])}")

        # 現在の環境との比較
        print("\n現在の環境:")
        print(f"- Python: {platform.python_version()}")
        print(f"- プラットフォーム: {platform.platform()}")

        # venv作成
        if create_venv(args.venv_path):
            # パッケージインストール
            if install_packages(args.venv_path, args.requirements):
                print("\n✓ 環境再構築が完了しました！")

                # 使用方法の案内
                if platform.system() == "Windows":
                    activate_cmd = f"{args.venv_path}\\Scripts\\activate"
                else:
                    activate_cmd = f"source {args.venv_path}/bin/activate"

                print(f"\n新しい環境を使用するには:")
                print(f"  {activate_cmd}")

                # CUDA互換性チェック
                check_cuda_compatibility(env_info, args.verbose)

                return 0

        return 1

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"エラー: 環境情報ファイルの読み込みに失敗しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
