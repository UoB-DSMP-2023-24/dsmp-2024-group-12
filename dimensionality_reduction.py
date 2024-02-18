import pandas as pd
import Embedding_module as eb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('vdj.csv')
id = df['complex.id']
gene = df['gene']
cdr = df['cdr3']
specificity = df['antigen.epitope']

amino_acids, amino_acid_to_index = eb.amino_acid_2_index(cdr)
cdr_one_hot = eb.one_hot_encode(cdr, amino_acids, amino_acid_to_index)
cdr_embedding = eb.embedding_encode(cdr_one_hot, amino_acids, 2, 10)
# print(cdr_embedding[0])
# print('-------------------------------')
# print(cdr_embedding[-1])

pca = PCA(n_components=2)
data = cdr_embedding.detach().numpy()
flattened_data = [sample.flatten() for sample in data]
transformed_data = pca.fit_transform(flattened_data)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
plt.show()