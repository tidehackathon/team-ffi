import pandas as pd
import ast


def string2list(string):
    if isinstance(string, str):
        x = ast.literal_eval(string)
        x = [n.strip() for n in x]
        return x
    else:
        return string

def remove_hashes(df_row):
    # print(df_row)
    content_string = df_row["content"]
    # print("content is type:", type(content_string))
    hashtags = df_row["hashtags"]
    if isinstance(hashtags, list):
        for ht in hashtags:
            # print(ht)
            # print("before:", content_string)
            content_string = content_string.replace("#"+ht, "")
            # print("after:", content_string)
            # print()
    # elif isinstance(hashtags, str):
    #     print("is a string:", hashtags)
    # else:
        # print(type(hashtags))
    return content_string

# print(remove_hashes({"content": "dette er en #test", "hashtags": "[\"test\"]"}))



DF = pd.read_pickle("/data/tide-hackaton/twitter_data/twitter_combined_df.pickle")

# Parse hashtag strings as lists
print("Parsing hashtag lists …")
DF["hashtags"] = DF["hashtags"].map(string2list)
print(type(DF["hashtags"].iloc[-1]))


# print(DF.loc[873022])
print("Removing hashtags …")
DF["content_wo_hashtags"] = DF.apply(remove_hashes, axis=1)
# test_df = DF.loc[873022:].apply(remove_hashes, axis=1)
# print(test_df)
# print(DF["content"])
DF.to_csv("/data/tide-hackaton/twitter_data/twitter_combined_df_wo_hashtags.csv")
DF.to_pickle("/data/tide-hackaton/twitter_data/twitter_combined_df_wo_hashtags.pickle")
# print(test_df[DF["content"] != test_df])

