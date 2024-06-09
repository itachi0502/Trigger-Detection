def process_conll_file(conll_path, txt_path, output_path):
    # 读取txt文件内容
    with open(txt_path, 'r', encoding='utf-8') as f:
        txt_content = f.read().strip()
    # 将文本内容按模块分割
    txt_modules = txt_content.split('\n\n')

    # 读取conll文件内容
    with open(conll_path, 'r', encoding='utf-8') as f:
        conll_content = f.read().strip()
    # 将conll内容按行分割
    conll_lines = conll_content.split('\n')

    # 遍历conll每行数据
    i = 0
    while i < len(conll_lines):
        line = conll_lines[i]
        if line == '':
            i += 1
            continue
        # 获取句子
        sentence = [line.split()[0]]
        # 获取label
        label = [line.split()[-1]]
        i += 1

        # 继续获取该句子的其他单词及其label
        while i < len(conll_lines) and conll_lines[i] != '':
            sentence.append(conll_lines[i].split()[0])
            label.append(conll_lines[i].split()[-1])
            i += 1

        # 将句子转为字符串
        sentence_str = ' '.join(sentence)

        # 在txt模块中查找与该句子匹配的相关句子
        found = False
        for txt_module in txt_modules:
            # 获取原文和相关句子
            txt_lines = txt_module.split('\n')
            txt_sentence = txt_lines[0]
            txt_related_sentence = txt_lines[1] if len(txt_lines) > 1 else ''
            # print(txt_related_sentence.split())
            # 判断该句子是否与原文匹配
            if sentence_str == txt_sentence:
                # 添加相关句子到conll数据集中
                new_label = ['B-X'] + ['B-X'] * (len(txt_related_sentence.split()))
                # conll_lines[i - 1] += f' <EOS>'
                conll_lines[i:i] = [f'{word} {lbl}' for word, lbl in
                                    zip(['<EOS>'] + txt_related_sentence.split(), new_label)]
                found = True
                break

        if not found:
            i += 1

    # 将处理后的conll内容保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(conll_lines))


process_conll_file('/home/wzl/project/kbner/kb/datasets/EN-English4/en_test.txt', '/home/wzl/project/kbner/kb/datasets/output_all3.txt', '/home/wzl/project/kbner/kb/datasets/EN-English4_conll_rank_eos_doc_full_wiki_v3_withent/test.txt')