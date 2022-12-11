import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data(file: str):
    df = pd.read_csv(file)
    return df

def compare_prune(df: pd.DataFrame, netwrok_names):
    means_prune, means_unprune = [], []
    for network in netwrok_names:
        prune = df[(df["network"]==network) & (df["method"]=="map") & (df["prune"]==True) & (df["heuristic"]=="fill")]
        unprune = df[(df["network"]==network) & (df["method"]=="map") & (df["prune"]==False) & (df["heuristic"]=="fill")]
        mean_prune = prune["time"].mean()
        mean_unprune = unprune["time"].mean()
        means_prune.append(mean_prune)
        means_unprune.append(mean_unprune)
        print(mean_prune)
        print(mean_unprune)
    return means_prune, means_unprune

def compare_heuristic(df: pd.DataFrame):
    fill = df[(df["network"]=="network-30") & (df["method"]=="mpe") & (df["prune"]==True) & (df["heuristic"]=="fill")]
    degree = df[(df["network"]=="network-30") & (df["method"]=="mpe") & (df["prune"]==True) & (df["heuristic"]=="degree")]
    mean_fill = fill["time"].mean()
    mean_degree = degree["time"].mean()
    print(mean_fill)
    print(mean_degree)
    return mean_fill, mean_degree

def compare_qe_length(df: pd.DataFrame):
    df = df[(df["network"]=="network-50") & (df["method"]=="mpe") & (df["prune"]==False) & (df["heuristic"]=="degree")]
    df["qe_length"] = df.apply(calculate_qe_len, axis=1)
    qe_len_set = set(df["qe_length"].to_list())
    qe_len_list = []
    mean_qe_len_list = []
    for qe_len in qe_len_set:
        qe_len_df = df[df["qe_length"]==qe_len]
        mean_qe_len = qe_len_df["time"].mean()
        print("len {} time: {}".format(qe_len, mean_qe_len))
        qe_len_list.append(qe_len)
        mean_qe_len_list.append(mean_qe_len)
    return qe_len_list, mean_qe_len_list

def calculate_qe_len(x):
    q_len = len(list(eval(x["query"])))
    e_len = len(eval(x["evidence"]))
    return q_len + e_len

def plot_double_bar(x, y1, y2):
    width = 0.4
    x2 = [one+width for one in x]
    plt.figure(figsize=(8,4))
    plt.bar(x=x, height=y1, width=width, label="Prune")
    plt.bar(x=x2, height=y2, width=width, label="Without Prune")
    for x_value, y_value in zip(x, y1):
        plt.text(x=x_value, y=y_value, s=y_value)
    for x_value, y_value in zip(x, y2):
        plt.text(x=x_value, y=y_value, s=y_value)
    plt.rcParams["axes.unicode_minus"] = False
    plt.title("Inference time with vs without netwrok pruning")
    plt.legend()
    plt.show()

file = "experiments/network-30.csv"
# compare_prune(load_data(file))
# compare_heuristic(load_data(file))
# compare_qe_length(load_data(file))

network_names = ["network-10", "network-20", "network-30"]
means_prune, means_unprune = compare_prune(load_data(file), network_names)
plot_double_bar([10, 20, 30], means_prune, means_unprune)