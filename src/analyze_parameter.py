import pandas as pd
import pingouin as pg

df = pd.read_csv('../simplified_data/results_parameters.csv')

result = df.groupby('model')['r2'].mean()

result_df = result.reset_index() 


result_df = result_df.set_index(['model','r2'])

result_df = result.reset_index() 

result_df['base_model'] = result_df['model'].str.split('_').str[0]

sorted_results = (
    result_df.groupby('base_model', group_keys=False)
    .apply(lambda group: group.sort_values(by='r2', ascending=True))
)


sorted_results.to_csv('../simplified_data/results_parameters_avg_sorted.csv', index=False)



# print(pg.rm_anova(data=df, dv='r2', within='model', subject='pid'))
# print(pg.pairwise_tests(dv='r2', within='model', subject='pid', data=df).round(3))