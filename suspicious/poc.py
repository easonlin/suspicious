import lda
import numpy as np
import pandas as pd


def load_data():
    columns = ["time", "tdur", "sip", "dip", "sport", "dport", "ipkt", "ibyt", "opkt", "obyt"]
    normal = [["2016-05-05 00:11:01", 0.03, "192.168.0.1", "192.168.0.2", 54651, 80, 2, 240, 3, 480]] * 9
    anomaly = [["2016-05-05 03:11:01", 0.03, "192.168.0.1", "192.168.0.2", 54651, 80, 2, 240, 3, 480]]
    df = pd.DataFrame(normal + anomaly, columns=columns)
    return df


def compute_deciles(array):
    return np.percentile(array, [10, 20, 30, 40, 50, 60, 70, 80, 90])


def compute_quintiles(array):
    return np.percentile(array, [20, 40, 60, 80])


def compute_word(df):
    dt = pd.to_datetime(df["time"])
    timeat = (dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second).values
    time_cut = compute_deciles(timeat)
    ipkt_cut = compute_deciles(df["ipkt"])
    ibyt_cut = compute_quintiles(df["ibyt"])
    w_time = np.digitize(timeat, time_cut, right=True)
    w_ipkt = np.digitize(df["ipkt"], ipkt_cut)
    w_ibyt = np.digitize(df["ibyt"], ibyt_cut)
    w_port = np.where((df["sport"] < df["dport"]) & (df["sport"] > 0) | (df["dport"] == 0), df['sport'], df['dport'])
    w_port = np.where((df["sport"] < 1024) | (df["dport"] < 1024), w_port, 333333)
    w_port = np.where((df["sport"] > 1024) | (df["dport"] > 1024) | (df["sport"] == 0) | (df["dport"] == 0), w_port,
                      111111)
    w_sdir = np.where((df["sport"] < 1024) & (df["sport"] > 0) & ((df["dport"] > 1024) | (df["dport"] == 0)), "-1_", "")
    w_ddir = np.where((df["dport"] < 1024) & (df["dport"] > 0) & ((df["sport"] > 1024) | (df["sport"] == 0)), "-1_", "")
    sword = pd.Series(w_sdir) + w_port.astype(str) + "_" + w_time.astype(str) + "_" + w_ibyt.astype(
        str) + "_" + w_ipkt.astype(str)
    dword = pd.Series(w_ddir) + w_port.astype(str) + "_" + w_time.astype(str) + "_" + w_ibyt.astype(
        str) + "_" + w_ipkt.astype(str)
    df_r = pd.DataFrame({"sip": df.sip, "dip": df.dip, "sword": sword, "dword": dword}, index=df.index)
    return df_r


def compute_ip_word_count(df):
    s_word_count = df.groupby(["sip", "sword"]).size()
    d_word_count = df.groupby(["dip", "dword"]).size()
    df_ip_word_count = pd.concat([s_word_count, d_word_count]).sum(level=[0, 1])
    df_ip_word_count.index.set_names(["ip", "word"], inplace=True)
    return df_ip_word_count


def form_topic_word(df_ip_word_count):
    df_topic_word = df_ip_word_count.unstack().fillna(0).astype(int)
    return df_topic_word


def train(df):
    X = df.as_matrix()
    model = lda.LDA(n_topics=20, n_iter=20, random_state=1)
    model.fit(X)  # model.fit_transform(X) is also available
    return model


def predict(model, df_word, df_topic_word, df_ip_word_count):
    X = df_topic_word.as_matrix()
    vocal = df_topic_word.columns.values
    titles = df_topic_word.index.values
    y = model.transform(X)
    ss_ip_topic = pd.Series(xrange(len(titles)), index=pd.Index(titles, name="ip"), name="ip_topic")
    ss_word_topic = pd.Series(xrange(len(vocal)), index=pd.Index(vocal, name="word"), name="word_topic")
    ix = df_ip_word_count.to_frame().join(ss_ip_topic).join(ss_word_topic)
    topic_word = model.topic_word_
    prob = np.einsum('ij,ij->j', y[ix.ip_topic].T, topic_word[:, ix.word_topic])
    prob = pd.Series(prob, index=df_ip_word_count.index, name="prob")
    df_prob = df_word[["sip", "sword", "dip", "dword"]].join(prob, on=["sip", "sword"]).join(prob, on=["dip", "dword"],
                                                                                             lsuffix="s", rsuffix="d")
    ss_prob = df_prob[["probs", "probd"]].max(axis=1)
    ss_prob.name = "prob"
    return ss_prob


def compute_result(df, df_word, ss_prob):
    df_r = df.join(df_word[["sword", "dword"]]).join(ss_prob)
    return df_r


def check(df_r):
    print np.array_equal(
        df_r.query("prob<0.1")[["time", "sip", "dip", "sport", "dport", "ipkt", "ibyt", "opkt", "obyt"]].values,
        np.array([['2016-05-05 03:11:01', '192.168.0.1', '192.168.0.2', 54651L, 80L, 2L, 240L, 3L, 480L]],
                 dtype=object))


df = load_data()
df_word = compute_word(df)
df_ip_word_count = compute_ip_word_count(df_word)
df_topic_word = form_topic_word(df_ip_word_count)
model = train(df_topic_word)
ss_prob = predict(model, df_word, df_topic_word, df_ip_word_count)
df_r = compute_result(df, df_word, ss_prob)
check(df_r)
