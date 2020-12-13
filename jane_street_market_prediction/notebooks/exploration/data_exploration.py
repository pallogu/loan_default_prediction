import pandas as pd
from pandas_profiling import ProfileReport
import pickle


train = pd.read_csv("../../data/raw/jane-street-market-prediction/train.csv")
features = pd.read_csv("../../data/raw/jane-street-market-prediction/features.csv")

train.info()

columns = train.columns.values

feats = ["feature_{count}".format(count = count) for count in range(0, 130)]

non_features = [column for column in columns if column not in feats]

non_features

train[['resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']].describe()

features

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
