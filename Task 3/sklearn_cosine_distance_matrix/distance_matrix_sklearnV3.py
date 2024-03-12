import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import Embedding_module as eb
import matplotlib.pyplot as plt
import seaborn as sns


#LOAD DATA
df = pd.read_csv('vdjdb1.1.csv')

df_alpha = df[df['gene'] == 'TRA']
df_beta = df[df['gene'] == 'TRB']

cdr_alpha = df_alpha['cdr3'].tolist()
cdr_beta = df_beta['cdr3'].tolist()

id = df['complex.id']
cdr_combined = []
for i in id.unique().tolist():
    if i != 0:
        index = df[df['complex.id'] == i].index.tolist()
        cdr_combined.append(df.at[index[0], 'cdr3'] + df.at[index[1], 'cdr3'])



#ENCODE, EMBED AND VECTORISE
cdr = df['cdr3'].tolist()
amino_acids, amino_acid_to_index = eb.amino_acid_2_index(cdr)

cdr_list = [cdr_alpha, cdr_beta, cdr_combined]
one_hot_list = []
for cdrs in cdr_list:
    one_hot = eb.one_hot_encode(cdrs, amino_acids, amino_acid_to_index)
    one_hot_list.append(one_hot)

embedding_list = []
for one_hots in one_hot_list:
    embedding = eb.embedding_encode(one_hots, amino_acids, 2, 10)
    embedding_list.append(embedding)

pca = PCA(n_components=10, svd_solver='full')
flattened_data_list = []
for datas in embedding_list:
    flattened_data = [sample.flatten() for sample in datas.detach().numpy()]
    flattened_data_list.append(flattened_data)

reduced_data_list = []
for flattened_datas in flattened_data_list:
    reduced_data = pca.fit_transform(flattened_datas)
    print(reduced_data.shape)
    reduced_data_list.append(reduced_data)



#COMPUTE PAIRWISE DISTANCE MATRICES
pairwise_distance_matrices = []
for reduced_data in reduced_data_list:
    distance_matrix_sklearn = cosine_distances(reduced_data)
    distance_matrix_sklearn = distance_matrix_sklearn.astype(np.float32)
    pairwise_distance_matrices.append(distance_matrix_sklearn)
    print('loading..')

#DISPLAY SAMPLE USING HEATMAP
df = pd.DataFrame(pairwise_distance_matrices[0]).sample(1000)
sns.heatmap(df)
plt.show()



# CONDENSE AND SAVE HALF OF THE SYMMETRICAL DISTANCE MATRICES
condensed_distance_matrices = []
for matrix in pairwise_distance_matrices:
    condensed_matrix = matrix[np.triu_indices(matrix.shape[0], k=1)]
    condensed_distance_matrices.append(condensed_matrix)
    print('loading...')

np.savez_compressed('condensed_pairwise_distance_matrices.npz', alpha=condensed_distance_matrices[0], beta=condensed_distance_matrices[1], combined=condensed_distance_matrices[2])
print('Saved to machine')


#LOAD CONDENSED DATA AND RECONSTRUCT DISTANCE MATRIX HALVES
npz_file = np.load('condensed_pairwise_distance_matrices.npz')
condensed_vector = npz_file['alpha']

N = int(np.sqrt(2 * len(condensed_vector) + 0.25) + 0.5)
pairwise_matrix = np.zeros((N, N), dtype=np.float32)
index = 0
for i in range(N):
    for j in range(i + 1, N):
        pairwise_matrix[i, j] = condensed_vector[index]
        index += 1

print(pairwise_matrix)