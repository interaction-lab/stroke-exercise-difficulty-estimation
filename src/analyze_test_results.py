import pandas as pd
import pingouin as pg

df = pd.read_csv('../simplified_data/results.csv')

print(df.groupby('model')['mse'].mean())
print(df.groupby('model')['r2'].mean())

# print(pg.rm_anova(data=df, dv='r2', within='model', subject='pid'))
# print(pg.pairwise_tests(dv='r2', within='model', subject='pid', data=df).round(3))