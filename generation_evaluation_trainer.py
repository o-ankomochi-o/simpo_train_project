#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenAI評価に基づく損失を組み込んだテキスト生成トレーナー
"""

import json
import os
import time
from typing import Dict, List, Literal, Union

import torch
import torch.nn.functional as F
import wandb
from openai import OpenAI
from trl import CPOTrainer


class GenerationEvaluationTrainer(CPOTrainer):
    """
    文章生成・評価・損失統合トレーナー

    OpenAIを使用して生成されたテキストを評価し、その結果を損失関数に組み込みます。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        training_args = kwargs["args"]

        # 独自損失関連のハイパーパラメータ
        self.simpo_gamma = getattr(training_args, "simpo_gamma", 0.8)
        self.loss_type = getattr(training_args, "loss_type", "sigmoid")

        # 評価重み付けパラメータ
        self.eval_loss_weight = getattr(training_args, "eval_loss_weight", 0.1)

        # 生成関連のハイパーパラメータ
        self.enable_generation = getattr(training_args, "enable_generation", True)
        self.generation_interval = getattr(training_args, "generation_interval", 25)
        self.generation_batch_size = getattr(training_args, "generation_batch_size", 2)

        # OpenAI評価関連のパラメータ
        self.enable_openai_eval = getattr(training_args, "openai_evaluation", False)
        self.openai_model = getattr(training_args, "openai_model", "gpt-4o-2024-08-06")
        self.openai_api_key = getattr(
            training_args, "openai_api_key", os.environ.get("OPENAI_API_KEY", "")
        )

        # OpenAIクライアントの初期化
        if self.enable_openai_eval and self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                print(f"OpenAI client initialized with model: {self.openai_model}")
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {str(e)}")
                self.enable_openai_eval = False
        else:
            self.enable_openai_eval = False
            if self.enable_openai_eval:
                print("OpenAI evaluation disabled: API key not provided")

        self.step_counter = 0

        # 評価スコアを保存する変数
        self.latest_eval_scores = {}
        self.steps_since_last_eval = 0

        # 生成と評価結果を保存するディレクトリ
        self.generation_dir = os.path.join(training_args.output_dir, "generations")
        os.makedirs(self.generation_dir, exist_ok=True)

        print(f"Initialized GenerationEvaluationTrainer with parameters:")
        print(f"  - simpo_gamma: {self.simpo_gamma}")
        print(f"  - loss_type: {self.loss_type}")
        print(f"  - eval_loss_weight: {self.eval_loss_weight}")
        print(f"  - generation_interval: {self.generation_interval}")

    # def log(self, logs, start_time=None):
    #     """
    #     拡張ログ機能 - super().log() を呼ばずに独自管理
    #     """
    #     print(f"📝 ログ記録発生！現在の global_step: {self.state.global_step}")
    #     print(f"logsに含まれるキー: {list(logs.keys())}")
    #     print(f"評価関連のキー: {[k for k in logs.keys() if k.startswith('eval_')]}")

    #     # 数値ログだけフィルタ
    #     numeric_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}

    #     # ローカルログ出力
    #     print(f"📊【ステップ {self.state.global_step}】WandBに送信するメトリクス一覧:")

    #     print("評価関連メトリクス検索中:")
    #     for k, v in numeric_logs.items():
    #         if k.startswith("eval_"):
    #             print(f"　🔹 評価メトリクス: {k}: {v}")
    #         else:
    #             print(f"　🔸 {k}: {v}")

    #     # wandb ログ
    #     if self.args.report_to == "wandb":
    #         wandb.log(numeric_logs, step=self.state.global_step)
    def log(self, logs, start_time=None):
        """
        拡張ログ機能 - wandbサポート付き+ 文字列系メトリクスを除外
        """
        # 数値メトリクスだけを渡す
        numeric_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        # 親クラスのlogメソッドを呼び出し
        super().log(numeric_logs)

        # 親クラスの処理後にwandbにも記録
        if self.args.report_to == "wandb":
            wandb.log(numeric_logs)

    def evaluate_with_openai(self, prompt, generated_text):
        """OpenAI APIを使用してテキストを評価する"""
        if not self.enable_openai_eval or not hasattr(self, "openai_client"):
            return {"error": "OpenAI evaluation not enabled"}

        try:
            # 評価指示
            evaluation_prompt = f"""
            評価してください:

            プロンプト:
            {prompt}

            生成されたテキスト:
            {generated_text}

            以下の項目について0〜5の数値で評価し、それぞれの理由も説明してください：
            1. 関連性: プロンプトの内容に関連しているか
            2. 多様性: 表現や語彙の多様性があるか
            3. 訴求点: 異なる観点や主張を含んでいるか
            4. 読みやすさ: 文章構造や流れの自然さ
            5. 全体評価: 総合的な質

            結果は以下のJSON形式で返してください:
            ```json
            {{
            "relevance": {{"score": 数値, "reason": "理由"}},
            "diversity": {{"score": 数値, "reason": "理由"}},
            "appeals": {{"score": 数値, "reason": "理由"}},
            "readability": {{"score": 数値, "reason": "理由"}},
            "overall": {{"score": 数値, "reason": "理由"}}
            }}
            """

            # APIリクエスト
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは文章評価の専門家です。指示に従って生成された文章を評価してください。",
                    },
                    {"role": "user", "content": evaluation_prompt},
                ],
                temperature=0,
            )

            response_text = response.choices[0].message.content.strip()
            response_text = response_text.strip("```json").strip("```").strip()
            print("json")
            print(response_text)

            # JSONの抽出
            try:
                # JSON部分の抽出
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    evaluation_result = json.loads(json_str)
                else:
                    # JSONが見つからない場合、全体をJSONとして解析
                    evaluation_result = json.loads(response_text)

                return evaluation_result
            except json.JSONDecodeError:
                print(f"Failed to parse OpenAI response as JSON: {response_text}")
                return {
                    "error": "Failed to parse response",
                    "raw_response": response_text,
                }

        except Exception as e:
            print(f"Error in OpenAI evaluation: {str(e)}")
            return {"error": str(e)}

    def extract_metrics_from_evaluation(self, evaluation_result):
        """OpenAI評価結果からスコアを抽出し、確実に数値型で返す"""
        metrics = {}

        # エラーチェック
        if "error" in evaluation_result:
            return {"evaluation_error": 1.0}

        try:
            score_keys = ["relevance", "diversity", "appeals", "readability", "overall"]
            score_sum = 0.0
            count = 0

            for key in score_keys:
                if key in evaluation_result and isinstance(
                    evaluation_result[key], dict
                ):
                    # スコア取得と数値変換を一度に
                    try:
                        score = float(evaluation_result[key].get("score", 0))
                        metrics[f"eval_{key}"] = score
                        metrics[f"eval_{key}_reason"] = evaluation_result[key].get(
                            "reason", ""
                        )
                        score_sum += score
                        count += 1
                    except (ValueError, TypeError):
                        pass  # 変換エラーは無視

            # 平均スコア計算
            if count > 0:
                metrics["eval_average_score"] = score_sum / count

            return metrics

        except Exception as e:
            return {"evaluation_parsing_error": 1.0}

    def compute_evaluation_loss(self):
        """評価スコアベースの損失を計算"""
        # 評価スコアがない場合はゼロ損失を返す
        if (
            not self.latest_eval_scores
            or "eval_average_score" not in self.latest_eval_scores
        ):
            return torch.tensor(0.0, device=self.model.device)

        # 評価スコアから損失を計算
        # スコアが高いほど損失が低くなるように負の符号をつける
        # スコアは通常0-5なので、適切にスケーリングする
        eval_score = self.latest_eval_scores.get("eval_average_score", 0)

        # スコアが高いほど損失が小さくなるように変換
        # 5点満点なら、5-score で0に近づくほど良いことになる
        eval_loss = (5.0 - eval_score) / 5.0

        scaled_loss = eval_loss

        return torch.tensor(scaled_loss, device=self.model.device)

    def generate_samples(self, model, batch):
        """バッチからサンプルを生成し、評価する関数"""
        # 小さいバッチを使用して効率化
        mini_batch = {
            "prompt_input_ids": batch["prompt_input_ids"][: self.generation_batch_size],
            "prompt_attention_mask": batch["prompt_attention_mask"][
                : self.generation_batch_size
            ],
        }

        # プロンプトのテキスト（あれば）を取得
        prompt_texts = []
        if "prompt" in batch:
            prompt_texts = batch["prompt"][: self.generation_batch_size]

        # 生成前に勾配チェックポイントを一時的に無効化
        was_gradient_checkpointing_enabled = False
        if (
            hasattr(model, "is_gradient_checkpointing")
            and model.is_gradient_checkpointing
        ):
            was_gradient_checkpointing_enabled = True
            model.gradient_checkpointing_disable()

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=mini_batch["prompt_input_ids"],
                    attention_mask=mini_batch["prompt_attention_mask"],
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.8,
                    num_beams=1,
                    pad_token_id=self.processing_class.pad_token_id,
                )

                decoded_texts = self.processing_class.batch_decode(
                    outputs, skip_special_tokens=True
                )
        finally:
            # 元の設定に戻す
            if was_gradient_checkpointing_enabled:
                model.gradient_checkpointing_enable()

        # 評価結果を保存するリスト
        evaluations = []
        evaluation_metrics = {}

        # プロンプトと生成文を表示
        print("\n===== 生成されたサンプル =====")
        for i, text in enumerate(decoded_texts):
            if i < len(prompt_texts):
                prompt = prompt_texts[i]
                response = text[len(prompt) :]
                print(f"[プロンプト {i}]: {prompt}")
                print(f"[生成応答 {i}]: {response}")
                print("-" * 80)

                # OpenAIによる評価（有効な場合）
                if self.enable_openai_eval:
                    print(f"OpenAI評価中... プロンプト {i}")
                    evaluation = self.evaluate_with_openai(prompt, response)
                    print("==============openaiの評価===============")
                    print(evaluation)
                    print("==========================================")
                    evaluations.append(
                        {
                            "prompt": prompt,
                            "response": response,
                            "evaluation": evaluation,
                        }
                    )

                    # メトリクスを抽出して集計
                    sample_metrics = self.extract_metrics_from_evaluation(evaluation)
                    for key, value in sample_metrics.items():
                        if key in evaluation_metrics:
                            evaluation_metrics[key].append(value)
                        else:
                            evaluation_metrics[key] = [value]

                    # 評価結果を表示
                    print(
                        f"評価結果: {json.dumps(evaluation, ensure_ascii=False, indent=2)}"
                    )
                    print("-" * 80)
            else:
                print(f"[生成全文 {i}]: {text}")
                print("-" * 80)

        # 評価結果をファイルに保存
        if self.enable_openai_eval and evaluations:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            eval_file = os.path.join(
                self.generation_dir, f"eval_step_{self.step_counter}_{timestamp}.json"
            )
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(evaluations, f, ensure_ascii=False, indent=2)
            print(f"評価結果を保存しました: {eval_file}")

        # 平均メトリクスを計算
        avg_metrics = {}
        for key, values in evaluation_metrics.items():
            if isinstance(values[0], (int, float)):  # 数値だけ平均する
                avg_metrics[key] = sum(values) / len(values)
            else:
                # 文字列（例：理由）などは最初の一つだけ代表で残す
                avg_metrics[key] = values[0]

        # 評価結果を保存
        self.latest_eval_scores = avg_metrics
        self.steps_since_last_eval = 0

        return decoded_texts, avg_metrics

    def get_batch_loss_metrics(
        self,
        model,
        batch,
        train_eval: Literal["train", "eval"] = "train",
    ):
        evaluation_metrics = {}

        # 一定間隔でサンプル生成と評価
        if (
            train_eval == "train"
            and self.enable_generation
            and self.step_counter % self.generation_interval == 0
        ):
            print(f"\n[Step {self.step_counter}] サンプル生成実行")
            generated_texts, evaluation_metrics = self.generate_samples(model, batch)
            print(f"生成されたサンプル数: {len(generated_texts)}")

        # 親クラス(CPOTrainer)の方法を呼び出す
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

        if train_eval == "train":
            # 評価スコアベースの損失を計算
            eval_loss = self.compute_evaluation_loss()

            # 損失に加算
            loss = loss + self.eval_loss_weight * eval_loss

            # メトリクスに追加
            metrics["evaluation_based_loss"] = eval_loss.item()

            # 評価からのステップ数を更新
            self.steps_since_last_eval += 1

        # 評価メトリクスを追加（同期処理を適用）
        print(f"評価メトリクスの内容: {evaluation_metrics}")  # デバッグ用
        if evaluation_metrics:
            # 新しい名前の規則を適用するためのプレフィックス
            prefix = "eval_" if train_eval == "eval" else ""

            # キーマッピングの定義
            key_mapping = {
                "eval_relevance": f"{prefix}openai/relevance",
                "eval_diversity": f"{prefix}openai/diversity",
                "eval_appeals": f"{prefix}openai/appeals",
                "eval_readability": f"{prefix}openai/readability",
                "eval_overall": f"{prefix}openai/overall",
                "eval_average_score": f"{prefix}openai/average_score",
            }

            # 評価メトリクスの処理（理由フィールド以外）
            for key, value in evaluation_metrics.items():
                if not key.endswith("_reason") and isinstance(value, (int, float)):
                    # 新しいキー名に変換
                    new_key = key_mapping.get(key, key)

                    # テンソル化して同期処理
                    tensor_value = torch.tensor(value, device=self.model.device)
                    metrics[new_key] = (
                        self.accelerator.gather_for_metrics(tensor_value).mean().item()
                    )

            # 評価ベースの損失も新しい命名規則で追加
            if train_eval == "train" and "evaluation_based_loss" in metrics:
                metrics[f"{prefix}openai/loss"] = metrics["evaluation_based_loss"]

        # ステップカウンタを更新
        if train_eval == "train":
            self.step_counter += 1

        print(f"最終的なmetrics辞書の内容: {metrics}")

        return loss, metrics
