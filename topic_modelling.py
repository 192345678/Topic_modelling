from transformers import DistilBertConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig,BertForSequenceClassification,BertConfig, BertTokenizer,TFBertForSequenceClassification
# 加载模型和分词器，并选择对应的配置类
import numpy as np
model = TFBertForSequenceClassification.from_pretrained("/", config=BertConfig.from_pretrained("../model/"))
tokenizer = BertTokenizer.from_pretrained("/")

# 示例用法：使用模型进行推理
input_text = "2024季后赛湖人以108:110憾负掘金止步首轮，詹姆斯空砍37分"
# input_text = "百度宣布发布文心一言：以达到GPT3.5的水平"
# input_text = "巴伊战争正式打响，美国宣布保持中立"
input_ids = tokenizer.encode(input_text, return_tensors="tf", padding=True, truncation=True)
output = model(input_ids)

# 打印推理结果
logits = output.logits  # 获取模型输出的各类别得分
predicted_class_index = logits.numpy().argmax()  # 获取最大概率类别的索引
print(predicted_class_index)

topic = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music","Family & Relationships", "Politics & Government"]
print(logits)
print(topic[predicted_class_index])