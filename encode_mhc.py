import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('trb_vdj_kmer.csv')
mhc_b = df['mhc.b'].tolist()



def binary_encoder(data, col):
    data_sort = data.sort_values(by = col)
    encoder = ce.BinaryEncoder(cols = [col], handle_missing = 'return_nan')
    dfbin = encoder.fit_transform(data_sort[col])
    data_sort = pd.concat([data_sort, dfbin], axis = 1)
    return data_sort


df = binary_encoder(df, 'mhc.a')
le = LabelEncoder()
df['mhc.b_label'] = le.fit_transform(mhc_b)

print(df.head())
