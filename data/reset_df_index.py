import pandas as pd

df = pd.read_pickle("/data/tide-hackaton/twitter_data/twitter_combined_df.pickle")

print(df)
df.reset_index(inplace=True, drop=True)
print(df)

df.to_pickle("/data/tide-hackaton/twitter_data/twitter_combined_df_reset_index.pickle")