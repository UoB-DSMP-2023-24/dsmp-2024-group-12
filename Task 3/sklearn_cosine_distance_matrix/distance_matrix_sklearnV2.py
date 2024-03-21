import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import encoding_96dim as en
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

cdr_list = [cdr_alpha, cdr_beta, cdr_combined]
'''


#ENCODE AND VECTORISE
vectorised_dataset_list = [[],[],[]]
for num, sequences in enumerate(cdr_list):
    for seq in sequences:
        vectorised_data = en.Encoding96(seq)
        vectorised_dataset_list[num].append(vectorised_data)
    vectorised_dataset_list[num] = np.array(vectorised_dataset_list[num])
    print(vectorised_dataset_list[num].shape)


#COMPUTE PAIRWISE DISTANCE MATRICES
pairwise_distance_matrices = []
for vectorised_dataset in vectorised_dataset_list:
    distance_matrix_sklearn = cosine_distances(vectorised_dataset)
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

print(pairwise_matrix)'''