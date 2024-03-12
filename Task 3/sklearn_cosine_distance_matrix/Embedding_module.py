import torch
import torch.nn as nn


def amino_acid_2_index(cdr3):
    amino_acids = set()
    for sequences in cdr3:
        amino_acids.update(sequences)
    # 创建氨基酸到索引的映射
    amino_acid_to_index = {amino_acid: i for i, amino_acid in enumerate(amino_acids)}
    return amino_acids, amino_acid_to_index


def one_hot_encode(cdr3, amino_acids, amino_acid_to_index):
    max_length = max(len(seq) for seq in cdr3)
    encoded_cdr3 = []
    for sequence in cdr3:
        encoding = torch.zeros([max_length, len(amino_acids)], dtype=torch.int)
        for i, amino_acid in enumerate(sequence):
            if amino_acid in amino_acid_to_index:
                encoding[i, amino_acid_to_index[amino_acid]] = 1
        encoded_cdr3.append(encoding)
    one_hot_sequences = torch.stack(encoded_cdr3)
    return one_hot_sequences

def embedding_encode(one_hot_sequences, amino_acids, dim, recudecd_dimensional):
    max_indices = torch.argmax(one_hot_sequences, dim=dim)
    embed = nn.Embedding(len(amino_acids), recudecd_dimensional)
    outputs = embed(max_indices)
    return outputs