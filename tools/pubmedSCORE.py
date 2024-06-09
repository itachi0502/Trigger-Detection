import bert_score
import numpy as np

# 定义BertScore模型和设备
import torch

bert_model = 'bert-base-uncased'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_best_sentence(original_text, related_sentences):
    # 使用BertScore计算相关度分数
    scores = bert_score.score([original_text] * len(related_sentences), related_sentences, model_type=bert_model, device=device)
    scores = np.array(scores[2]) # 取出F1分数
    max_index = scores.argmax() # 找到分数最高的语句
    best_sentence = related_sentences[max_index]
    max_score = scores[max_index]
    return best_sentence, max_score

# 读取原始文本和相关语句
with open('/home/wzl/project/kbner/kb/datasets/all3.txt', 'r') as f:
    input_text = f.read()

# 按照模块划分
modules = input_text.strip().split('\n\n')
output_text = ""

for module in modules:
    lines = module.strip().split('\n')
    original_text = lines[0].strip()
    related_sentences = lines[1:]
    if len(related_sentences) == 0:
        # 如果没有相关语句，则跳过
        continue
    print(original_text)
    best_sentence, max_score = find_best_sentence(original_text, related_sentences)
    print(best_sentence+'\n')
    output_text += f"{original_text}\n{best_sentence}\n\n"

# 将结果保存到文件
with open('/home/wzl/project/kbner/kb/datasets/output_all3.txt', 'w') as f:
    f.write(output_text)
