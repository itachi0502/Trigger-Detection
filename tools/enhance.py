import random
import nltk
from nltk.corpus import wordnet
from transformers import pipeline

def read_conll(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split('\t')
            id_, label, text = parts
            data.append((id_, int(label), text))
    # print(data)
    return data

def write_data_to_file(data, output_filename):
    with open(output_filename, 'w') as outfile:
        for item in data:
            question_id = item[0]
            label = item[1]
            text = item[2]
            outfile.write(f"{question_id}\t{label}\t{text}\n")

# 随机调换句子中词的顺序
def random_shuffle(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)

# 反译
def back_translation(text, source_lang='en', target_lang='fr'):
    # 指定模型
    model_name = 't5-base'
    translator = pipeline('translation_{0}_to_{1}'.format(source_lang, target_lang), model=model_name, max_length=512)
    translated_text = translator(text)[0]['translation_text']
    translator_back = pipeline('translation_{0}_to_{1}'.format(target_lang, source_lang), model=model_name, max_length=512)
    back_translated_text = translator_back(translated_text)[0]['translation_text']
    # 后处理，删除重复的句子
    back_translated_text = ' '.join(list(dict.fromkeys(back_translated_text.split())))
    # 如果句子以逗号开头，删除逗号
    if back_translated_text.startswith(','):
        back_translated_text = back_translated_text[1:].strip()
    return back_translated_text


# 同义词替换
def synonym_replacement(text):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in nltk.corpus.stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= 1:
            break

    sentence = ' '.join(new_words)
    return sentence

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

# def augment_data(data):
#     augmented_data = []
#     for id_, label, text in data:
#         # 反译
#         print("text:" + text)
#         back_translated_text = back_translation(text)
#         print("back:"+back_translated_text)
#         # 将原文和反译后的译文通过[EOS]标签进行拼接
#         combined_text = text + " [EOS] " + back_translated_text
#         print("combined:"+combined_text)
#         augmented_data.append((id_, label, combined_text))
#
#     return augmented_data

def augment_data(data):
    augmented_data = []
    for id_, label, text in data:
        # # 随机调换句子中词的顺序
        print("text:"+text)
        shuffled_text = random_shuffle(text)
        print("suffled:"+shuffled_text)
        combined_text = text + " [EOS] " + shuffled_text
        print("combine_text:"+combined_text)
        augmented_data.append((id_, label, combined_text))

        # 反译
        # print("text:" + text)
        # back_translated_text = back_translation(text)
        # print("back:"+back_translated_text)
        # augmented_data.append((id_, label, back_translated_text))

        # # 同义词替换
        # print("text:"+text)
        # replaced_text = synonym_replacement(text)
        # print("replaced_text:"+replaced_text)
        # combined_text = text + " [EOS] " + replaced_text
        # print("combine_text:"+combined_text)
        # augmented_data.append((id_, label, combined_text))

    return augmented_data

data = read_conll('/home/wzl/project/kbner/kb/datasets/iron/new_train.txt')
augmented_data = augment_data(data)
print(augmented_data)
write_data_to_file(augmented_data, "/home/wzl/project/kbner/kb/datasets/iron_shuffled/train_data.txt")