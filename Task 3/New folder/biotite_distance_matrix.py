import pandas as pd
import numpy as np
import biotite.sequence.align as bio
from biotite.sequence import ProteinSequence


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

# Use list comprehension to create a list of objects
protein_sequence_list = [ProteinSequence(cdr3) for cdr3 in cdr_alpha]

matrix = bio.SubstitutionMatrix.std_protein_matrix()

alignment, order, tree, distances = bio.align_multiple(protein_sequence_list, matrix)

#print(alignment)
#print(distances)

np.savez_compressed('pairwise_distance_matrix.npz', alpha=distances)
#print('Saved to machine')