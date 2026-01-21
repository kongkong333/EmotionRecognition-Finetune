import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Qwen2Tokenizer
from transformers import Qwen2ForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda:0"

def demo():
    model_path = '/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct-emotext/checkpoint-2208'
    estimator : Qwen2ForCausalLM= AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer : Qwen2Tokenizer = AutoTokenizer.from_pretrained('/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct')

    system = """你是情绪分析专家，请从[生气,厌恶,恐惧,焦虑,难过,尴尬,失望,无聊,平静,好奇,困惑,吃惊,怀旧,满足,敬畏,钦佩,开心,兴奋]中选出最可能的3种情绪，评估其概率（概率和为1），以JSON格式输出：
    {"score": [{"emotion": "恐惧", "score": 0.45}, {"emotion": "焦虑", "score": 0.35}, {"emotion": "吃惊", "score": 0.20}]}"""

    while True:
        comment = input('请输入内容:')
        message = [{'role': 'system', 'content': system}, {'role': 'user', 'content': comment}]
        inputs = tokenizer.apply_chat_template(message,
                                               add_generation_prompt=True,
                                               tokenize=True,
                                               return_tensors='pt',
                                               return_dict=True).to(device)
        inputs_length = len(inputs['input_ids'][0])
        with torch.no_grad():
            outputs = estimator.generate(**inputs, max_length=512)
        output = outputs[0]
        y_pred = tokenizer.decode(output[inputs_length:], skip_special_tokens=True).strip()
        print('预测标签:', y_pred)
        print('-' * 50)


if __name__ == '__main__':
    demo()