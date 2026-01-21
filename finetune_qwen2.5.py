import torch
import pickle
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import Qwen2Tokenizer
from transformers import Qwen2ForCausalLM
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers.trainer_callback import TrainerCallback
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
random.seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct-emotext"
DEVICE = "cuda:0"


class LossCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.train_steps = []
        self.val_loss = []
        self.val_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次日志记录时触发"""
        if logs is None:
            return
        if "loss" in logs and state.global_step is not None:
            self.train_loss.append(logs["loss"])
            self.train_steps.append(state.global_step)
        if "eval_loss" in logs and state.global_step is not None:
            self.val_loss.append(logs["eval_loss"])
            self.val_steps.append(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_steps, self.train_loss, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(self.val_steps, self.val_loss, label='Validation Loss', color='red', alpha=0.7)

        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        loss_plot_path = os.path.join(args.output_dir, "train_val_loss_step_curve.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练/验证Loss曲线图已保存至：{loss_plot_path}")

def get_dataset(tokenizer):
    """
    读取JSONL格式的情绪分析数据集，并转换为模型训练所需的格式
    """
    result_data = []
    jsonl_file_path = "/root/emotion_rec/finetune_data.jsonl"

    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                message = [
                    {'role': 'system', 'content': data['instruction']},
                    {'role': 'user', 'content': data['input']},
                    {'role': 'assistant', 'content': json.dumps(data['output'], ensure_ascii=False)}
                ]

                inputs = tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=False,
                    tokenize=True
                )
                result_data.append(inputs)
    except FileNotFoundError:
        print(f"未找到文件 {jsonl_file_path}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败，行内容：{line}，错误信息：{e}")
        exit(1)
    train_data, val_data = train_test_split(
        result_data,
        test_size=0.05,
        random_state=42
    )
    print(f"成功加载 {len(result_data)} 条数据")
    return train_data, val_data

def demo():
    estimator : Qwen2ForCausalLM= AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": DEVICE},
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer : Qwen2Tokenizer = AutoTokenizer.from_pretrained(model_path)

    arguments = TrainingArguments(output_dir=OUTPUT_DIR,
                                  per_device_train_batch_size=1,
                                  per_device_eval_batch_size=1,
                                  optim='adamw_torch',
                                  num_train_epochs=3,
                                  learning_rate=2e-5,
                                  eval_strategy='steps',
                                  eval_steps=5,
                                  save_strategy='epoch',
                                  logging_strategy='steps',
                                  logging_steps=5,
                                  gradient_accumulation_steps=4,
                                  save_total_limit=5,
                                  load_best_model_at_end=False,
                                  bf16=True
                                  )

    train_data, val_data = get_dataset(tokenizer)
    loss_callback = LossCallback()
    trainer = Trainer(model=estimator,
                      train_dataset=train_data,
                      eval_dataset=val_data,
                      args=arguments,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                      callbacks=[loss_callback]
                      )

    trainer.train()


if __name__ == '__main__':
    demo()