import pandas as pd
import torch
import torch.nn as nn




# 读取CSV文件
df = pd.read_csv('vdj.csv')

# 获取CDR3列的所有序列
cdr3 = df['cdr3']

amino_acids = set()
for sequences in cdr3:
    amino_acids.update(sequences)

# 创建氨基酸到索引的映射
amino_acid_to_index = {amino_acid: i for i, amino_acid in enumerate(amino_acids)}
max_length = max(len(seq) for seq in cdr3)
# eg = cdr3[0]
# encoding = torch.zeros([max_length, len(amino_acids)], dtype=torch.int)
# for i, amino_acid in enumerate(eg):
#     if amino_acid in amino_acid_to_index:
#         encoding[i, amino_acid_to_index[amino_acid]] = 1

# One-Hot编码函数
def one_hot_encode(sequence, max_length):
    encoding = torch.zeros([max_length, len(amino_acids)], dtype=torch.int)
    for i, amino_acid in enumerate(sequence):
        if amino_acid in amino_acid_to_index:
            encoding[i, amino_acid_to_index[amino_acid]] = 1
    return encoding

# 对每个 CDR3 序列进行 One-Hot 编码和填充
encoded_cdr3 = [one_hot_encode(seq, max_length) for seq in cdr3]
one_hot_sequences = torch.stack(encoded_cdr3)
max_indices = torch.argmax(one_hot_sequences, dim=2)


embed = nn.Embedding(len(amino_acids),10)
outputs = embed(max_indices)
print(outputs[0])
print("------------------")
print(outputs[1])
# # inputs = torch.LongTensor([[amino_acid_to_index[amino_acid] for amino_acid in eg]])
# inputs = torch.tensor(encoding)
# outputs = embed(torch.argmax(inputs, dim=1))
# print(outputs)
# # print(embed.weight)


