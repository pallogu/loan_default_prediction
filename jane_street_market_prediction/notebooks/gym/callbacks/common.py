import pandas as pd
import numpy as np

def calculate_u_metric(df, model, verbose=0):
    actions = model.predict(df[[c for c in df.columns if "f_" in c] + ["feature_0", "weight"]].values,
                            deterministic=True)[0]
    assert not np.isnan(np.sum(actions))

    sum_of_actions = np.sum(actions)

    df["action"] = pd.Series(data=actions, index=df.index)

    df["trade_reward"] = df["action"] * df["weight"] * df["resp"]

    tmp = df.groupby(["date"])[["trade_reward"]].agg("sum")

    sum_of_pi = tmp["trade_reward"].sum()
    sum_of_pi_x_pi = (tmp["trade_reward"] * tmp["trade_reward"]).sum()
    if sum_of_pi_x_pi == 0:
        return -1000, 0, 0

    t = sum_of_pi / np.sqrt(sum_of_pi_x_pi) * np.sqrt(250 / tmp.shape[0])
    u = np.min([np.max([t, 0]), 6]) * sum_of_pi
    ratio_of_ones = sum_of_actions / len(actions)

    if verbose == 1:
        print("sum of pi: {sum_of_pi}".format(sum_of_pi=sum_of_pi))
        print("t: {t}".format(t=t))
        print("u: {u}".format(u=u))
        print("np_sum(actions)", sum_of_actions)
        print("ration of ones", ratio_of_ones)
        print("length of df", len(actions))

    return t, u, ratio_of_ones