import pandas as pd
import category_encoders as ce

df = pd.read_csv('subvdj.csv')
# df = df.fillna('0')

# alpha beta链子表
df_alpha = df[df['gene'] == 0]
df_beta = df[df['gene'] == 1]

def binary_encoder(data, col):
    data_sort = data.sort_values(by = col)
    encoder = ce.BinaryEncoder(cols = [col], handle_missing = 'return_nan')
    dfbin = encoder.fit_transform(data_sort[col])
    data_sort = pd.concat([data_sort, dfbin], axis = 1)
    return data_sort

col_list = ['v.seg', 'd.segm', 'j.segm']
for cols in col_list:
    df_alpha = binary_encoder(df_alpha, cols)


# df_alpha_sort = df_alpha.sort_values(by='v.seg')
# encoder = ce.BinaryEncoder(cols=['v.seg'], handle_missing = 'return_nan')
# dfbin = encoder.fit_transform(df_alpha_sort['v.seg'])
# df_alpha_sort = pd.concat([df_alpha_sort,dfbin],axis=1)

df_alpha.to_csv('alpha_encode_vdj.csv', index=False)

