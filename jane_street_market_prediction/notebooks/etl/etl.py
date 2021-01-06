# +
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import pickle

from ETLc import ETL_1, ETL_2
# -

train = pd.read_csv("../../input/train.csv")

valuation_data = train[train["date"] >=400]
train = train[train["date"] < 400]

feats = ["feature_{count}".format(count = count) for count in range(1, 130)]
rest_cols = [column for column in train.columns if column not in feats]

train_mean = train.mean()
train_stddev = train.std()

etl_1 = ETL_1(
    initial_row=train_mean,
    mean = train_mean,
    stddev=train_stddev,
    columns_to_transform=feats,
)

# %%time
train_trans_1  = train.apply(etl_1.fillna_normalize, axis=1)

pca = PCA(n_components=0.95)
pca.fit(train_trans_1[feats].values)

etl_2 = ETL_2(
    columns_to_transform=feats,
    trans_cols_names= ["f_{i}".format(i=i) for i in range(40)],
    columns_to_keep_train = ['date', 'weight'],
    pca = pca
)

# %%time
train_trans_2  = train_trans_1.apply(etl_2.reduce_columns_train, axis=1)

train_trans_2

train_trans_2.to_csv("./train_dataset_after_pca.csv", index=False)

with open("./etl_1.pkl", "wb") as f:
    pickle.dump(etl_1, f)


with open("./etl_2.pkl", "wb") as f:
    pickle.dump(etl_2, f)



val_trans_1 = valuation_data.apply(etl_1.fillna_normalize, axis=1)

val_trans_2 = val_trans_1.apply(etl_2.reduce_columns_train, axis=1)

val_trans_2.to_csv("./val_dataset_after_pca.csv", index=False)

val_trans_2[val_trans_2["date"] < 420
           ].shape

# +
# %%time

test_df = pd.DataFrame(
    data=[
        {
            "feat_0": None,
            "feat_1": 2,
            "feat_2": 3
        },
        {
            "feat_0": 2,
            "feat_1": 1,
            "feat_2": 3
        },
        {
            "feat_0": 1,
            "feat_1": 1,
            "feat_2": 3
        },
        {
            "feat_0": None,
            "feat_1": 1,
            "feat_2": 3
        },
        {
            "feat_0": 1,
            "feat_1": None,
            "feat_2": 3
        }
    ],
    columns = ["feat_0", "feat_1", "feat_2"]
)

test_mean = test_df.mean()
test_std = test_df.std()

class TestPCA():
    def transform(self, row):
        a = np.array([[1, 2], [2, 1], [1, 1]])
        return np.matmul(a, row)
test_pca = TestPCA()


etl = ETL(
    initial_row = test_mean,
    columns_to_transform=["feat_0", "feat_1"],
    trans_cols_names=["f_{i}".format(i=i) for i in range(3)],
    columns_to_keep=["feat_2"],
    mean=test_mean,
    stddev=test_std,
    pca=test_pca
)

test_df.apply(etl.transform, axis=1)
# -

