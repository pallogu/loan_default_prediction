import pandas as pd
# from pandas_profiling import ProfileReport
import pickle



train = pd.read_csv("../../input/train.csv")
features = pd.read_csv("../../input/features.csv")

test = pd.read_csv("../../input/example_test.csv")

test.head()



train.shape

train[train["weight"] == 0 ]

# + active=""
#
# -



train.info()

columns = train.columns.values

feats = ["feature_{count}".format(count = count) for count in range(0, 130)]

non_features = [column for column in columns if column not in feats]

non_features

train[['resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']].describe()

train[train["resp"] >0].shape[0]/train.shape[0]

train[train["resp"] >0][["resp"]].describe()

train[train["resp"] < 0][["resp"]].describe()

1.523521/1.466193



train[feats[:3] + ["date", "resp", "ts_id"]][:10].values

features[:3] + ["date", "resp", "ts_id"]

train['date']

train[features].describe()

sample= train[features].sample(frac=0.05)

profile = ProfileReport(sample, title="Pandas Profiling Report")

profile

with open("profile.pkl", "wb") as f:
    pickle.dump(profile, f, pickle.HIGHEST_PROTOCOL)
    f.close()

profile.to_file("features.html")
