#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenAI評価に基づく損失を組み込んだテキスト生成トレーナー
"""

import json
import math
import os
import time
from typing import Dict, List, Literal, Union

import torch
import torch.nn.functional as F
import wandb
from openai import OpenAI
from trl import KTOTrainer


class KTOGenerationEvaluationTrainer(KTOTrainer):
    """
    OpenAI評価付きKTOトレーナー

    OpenAIを使用して生成されたテキストを評価し、
    その結果をKTOベースの損失関数に組み込みます。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        training_args = kwargs["args"]

        # SimPO/CPO特有のパラメータは原則削除して問題なし
        self.loss_type = getattr(training_args, "loss_type", "sigmoid")  # 必要なら保持

        # 評価重み
        self.eval_loss_weight = getattr(training_args, "eval_loss_weight", 0.1)

        # 生成設定
        self.enable_generation = getattr(training_args, "enable_generation", True)
        self.generation_interval = getattr(training_args, "generation_interval", 25)
        self.generation_batch_size = getattr(training_args, "generation_batch_size", 2)

        # OpenAI設定
        self.enable_openai_eval = getattr(training_args, "openai_evaluation", False)
        self.openai_model = getattr(training_args, "openai_model", "gpt-4o-2024-08-06")
        self.openai_api_key = getattr(
            training_args, "openai_api_key", os.environ.get("OPENAI_API_KEY", "")
        )

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
        self.latest_eval_scores = {}
        self.steps_since_last_eval = 0
        self.generation_dir = os.path.join(training_args.output_dir, "generations")
        os.makedirs(self.generation_dir, exist_ok=True)

        print(f"Initialized KTOGenerationEvaluationTrainer with parameters:")
        print(f"  - eval_loss_weight: {self.eval_loss_weight}")
        print(f"  - generation_interval: {self.generation_interval}")

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
        """OpenAI APIを使用してテキストを多次元評価する"""
        if not self.enable_openai_eval or not hasattr(self, "openai_client"):
            return {"error": "OpenAI evaluation not enabled"}

        evaluation_prompt = f"""
        以下のプロンプトと生成されたキャッチフレーズを読み取り、
        0〜5の数値でそれぞれ評価し、理由も返してください。

        ### プロンプト:
        {prompt}

        ### 生成されたキャッチフレーズ:
        {generated_text}

        ### 出力形式（JSONのみ）:
        ```json
        {{
          "relevance":   {{ "score": 数値, "reason": "..." }},
          "diversity":   {{ "score": 数値, "reason": "..." }},
          "appeals":     {{ "score": 数値, "reason": "..." }},
          "readability": {{ "score": 数値, "reason": "..." }},
          "overall":     {{ "score": 数値, "reason": "..." }}
        }}
        ```
        """

        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "あなたは文章評価の専門家です。"},
                {"role": "user", "content": evaluation_prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        # JSON 部分だけ取り出してパース
        json_str = text[text.find("{") : text.rfind("}") + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Failed to parse response", "raw": text}

    def extract_metrics_from_evaluation(self, evaluation_result):
        """多次元評価結果からスコアを抽出し、平均スコアも計算"""
        metrics = {}
        if "error" in evaluation_result:
            return {"evaluation_error": 1.0}

        score_keys = ["relevance", "diversity", "appeals", "readability", "overall"]
        total, count = 0.0, 0

        for key in score_keys:
            info = evaluation_result.get(key)
            if isinstance(info, dict):
                try:
                    s = float(info.get("score", 0))
                    metrics[f"eval_{key}"] = s
                    metrics[f"eval_{key}_reason"] = info.get("reason", "")
                    total += s
                    count += 1
                except (ValueError, TypeError):
                    pass

        if count > 0:
            metrics["eval_average_score"] = total / count
        return metrics

    def compute_evaluation_loss(self):
        """平均評価スコアから損失を計算"""
        avg = self.latest_eval_scores.get("eval_average_score")
        if avg is None:
            return torch.tensor(0.0, device=self.model.device)

        # 5点満点 → (5 - 平均)/5 で [0,1] にスケーリング
        loss_val = (5.0 - avg) / 5.0
        return torch.tensor(loss_val, device=self.model.device)

    def generate_samples(self, model, batch):
        """バッチからサンプルを生成し、評価する関数"""
        # 小さいバッチを使用して効率化
        mini_batch = {
            # "prompt_input_ids": batch["prompt_input_ids"][: self.generation_batch_size],
            "prompt_input_ids": batch["prompt_input_ids"],
            # "prompt_attention_mask": batch["prompt_attention_mask"][
            #     : self.generation_batch_size
            # ],
            "prompt_attention_mask": batch["prompt_attention_mask"],
        }

        # プロンプトのテキスト（あれば）を取得
        prompt_texts = []
        if "prompt" in batch:
            # prompt_texts = batch["prompt"][: self.generation_batch_size]
            prompt_texts = batch["prompt"]

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
                    max_new_tokens=256,
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
    # 1) 生成＆評価（multi‐metric）を呼び出す
    if (
        train_eval == "train"
        and self.enable_generation
        and self.step_counter % self.generation_interval == 0
    ):
        print(f"\n[Step {self.step_counter}] サンプル生成実行")
        generated_texts, _ = self.generate_samples(model, batch)
        print(f"生成されたサンプル数: {len(generated_texts)}")

    # 2) 通常の KTOTrainer 損失＋メトリクス取得
    loss, metrics = super().get_batch_loss_metrics(model, batch)

    # 3) 評価ベースの損失を追加
    if train_eval == "train":
        eval_loss = self.compute_evaluation_loss()
        loss = loss + self.eval_loss_weight * eval_loss
        metrics["evaluation_based_loss"] = eval_loss.item()
        self.steps_since_last_eval += 1

    # 4) multi‐metric の評価値をメトリクスに登録
    #    キー名を 1 のフォーマットに合わせてマッピング
    prefix = "eval_" if train_eval == "eval" else ""
    key_mapping = {
        "eval_relevance": f"{prefix}openai/relevance",
        "eval_diversity": f"{prefix}openai/diversity",
        "eval_appeals": f"{prefix}openai/appeals",
        "eval_readability": f"{prefix}openai/readability",
        "eval_overall": f"{prefix}openai/overall",
        "eval_average_score": f"{prefix}openai/average_score",
    }

    for raw_key, value in self.latest_eval_scores.items():
        # 数値スコアのみ登録
        if raw_key in key_mapping and isinstance(value, (int, float)):
            tensor_value = torch.tensor(value, device=self.model.device)
            # accelerator で複数GPU集約
            metrics[key_mapping[raw_key]] = (
                self.accelerator.gather_for_metrics(tensor_value).mean().item()
            )

    # 5) ステップカウンタ更新
    if train_eval == "train":
        self.step_counter += 1

    return loss, metrics
