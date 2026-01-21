# EmotionRecognition-Finetune
基于 Qwen2.5-0.5B-Instruct 模型，对自定义的 JSONL 格式情绪识别数据集进行全量指令微调，输出JSON格式的多类情绪识别结果。同时，训练时使用回调实时记录训练和验证损失并保存Loss曲线图。

## 环境核心配置
python==3.10.19
torch==2.9.1
transformers==4.57.6
peft==0.18.1
accelerate==1.12.0

## JSONL输入数据示例
'''JSON
{"instruction": "你是情绪分析专家，请从[生气,厌恶,恐惧,焦虑,难过,尴尬,失望,无聊,平静,好奇,困惑,吃惊,怀旧,满足,敬畏,钦佩,开心,兴奋]中选出最可能的3种情绪，评估其概率（概率和为1），以JSON格式输出：\n    {\"score\": [{\"emotion\": \"恐惧\", \"score\": 0.45}, {\"emotion\": \"焦虑\", \"score\": 0.35}, {\"emotion\": \"吃惊\", \"score\": 0.20}]}", "input": "喂！这好歹也是个手术啊！", "output": {"score": [{"emotion": "生气", "score": 0.62}, {"emotion": "平静", "score": 0.32}, {"emotion": "吃惊", "score": 0.06}]}}
{"instruction": "你是情绪分析专家，请从[生气,厌恶,恐惧,焦虑,难过,尴尬,失望,无聊,平静,好奇,困惑,吃惊,怀旧,满足,敬畏,钦佩,开心,兴奋]中选出最可能的3种情绪，评估其概率（概率和为1），以JSON格式输出：\n    {\"score\": [{\"emotion\": \"恐惧\", \"score\": 0.45}, {\"emotion\": \"焦虑\", \"score\": 0.35}, {\"emotion\": \"吃惊\", \"score\": 0.20}]}", "input": "你能不能不要说得跟玩一样？", "output": {"score": [{"emotion": "生气", "score": 0.68}, {"emotion": "厌恶", "score": 0.23}, {"emotion": "平静", "score": 0.09}]}}

