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
sub_data = [df_alpha, df_beta]
names = ['alpha_vdj_encode.csv', 'beta_vdj_encode.csv']
for sub_df, name in zip(sub_data, names):
    new_df = sub_df
    for cols in col_list:
        new_df = binary_encoder(new_df, cols)
    new_df.to_csv(name, index=False)


# df_alpha_sort = df_alpha.sort_values(by='v.seg')
# encoder = ce.BinaryEncoder(cols=['v.seg'], handle_missing = 'return_nan')
# dfbin = encoder.fit_transform(df_alpha_sort['v.seg'])
# df_alpha_sort = pd.concat([df_alpha_sort,dfbin],axis=1)


