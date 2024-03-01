import pandas as pd
import umap
import Embedding_module as eb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import manifold,datasets

df = pd.read_csv('vdj.csv')

# alpha beta链子表
df_alpha = df[(df['gene'] == 'TRA') & (df['species'] == 'HomoSapiens')]
df_beta = df[(df['gene'] == 'TRB') & (df['species'] == 'HomoSapiens')]

# alpha beta链的cdr3和specificity
cdr_alpha = df_alpha['cdr3'].tolist()
cdr_beta = df_beta['cdr3'].tolist()
specificity_alpha = df_alpha['antigen.gene'].tolist()
specificity_beta = df_beta['antigen.gene'].tolist()

# combined链的cdr3和specificity
id = df[df['species'] == 'HomoSapiens']['complex.id']
cdr_combined = []
specificity_combined = []
for i in id.unique().tolist():
    index = df[df['complex.id'] == i].index.tolist()
    cdr_combined.append(df.at[index[0], 'cdr3'] + df.at[index[1], 'cdr3'])
    specificity_combined.append(df.at[index[0], 'antigen.gene'])

# 创建氨基酸到索引的映射
cdr = df['cdr3'].tolist()
amino_acids, amino_acid_to_index = eb.amino_acid_2_index(cdr)


# 计算三种chain的one-hot
cdr_list = [cdr_alpha, cdr_beta, cdr_combined]
one_hot_list = []
for cdrs in cdr_list:
    one_hot = eb.one_hot_encode(cdrs, amino_acids, amino_acid_to_index)
    one_hot_list.append(one_hot)


# 计算三种chain的embedding
embedding_list = []
for one_hots in one_hot_list:
    embedding = eb.embedding_encode(one_hots, amino_acids, 2, 10)
    embedding_list.append(embedding)


# 三种chain的specificity的编码
specificity_list = [specificity_alpha, specificity_beta, specificity_combined]
le = LabelEncoder()
numeric_specificity_list = []
for specificities in specificity_list:
    numeric_specificity = le.fit_transform(specificities)
    numeric_specificity_list.append(numeric_specificity)

Umap = umap.UMAP(n_components=2)
tsne = manifold.TSNE(n_components=2, init='pca', random_state=1, learning_rate=100)
pca = PCA(n_components=2, svd_solver='full')
# 展平数据
flattened_data_list = []
for datas in one_hot_list:
    flattened_data = [sample.flatten() for sample in datas.detach().numpy()]
    flattened_data_list.append(flattened_data)
# 降维
reduced_data_list = []
for flattened_datas in flattened_data_list:
    # UMAP
    reduced_data = Umap.fit_transform(flattened_datas)
    # tSNE
    # reduced_data = tsne.fit_transform(flattened_datas)
    # PCA
    # reduced_data = pca.fit_transform(flattened_datas)
    reduced_data_list.append(reduced_data)


plt.subplot(1, 3, 1)
plt.scatter(reduced_data_list[0][:, 0], reduced_data_list[0][:, 1], c = numeric_specificity_list[0], cmap='Spectral',
            marker='.', s = 10)

plt.subplot(1, 3, 2)
plt.scatter(reduced_data_list[1][:, 0], reduced_data_list[1][:, 1], c = numeric_specificity_list[1], cmap='Spectral',
            marker='.', s = 10)

plt.subplot(1, 3, 3)
plt.scatter(reduced_data_list[2][:, 0], reduced_data_list[2][:, 1], c = numeric_specificity_list[2], cmap='Spectral',
            marker='.', s = 10)

# plt.tight_layout()
plt.show()