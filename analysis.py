import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data(file: str):
    df = pd.read_csv(file)
    return df

def compare_prune(df: pd.DataFrame, network_names, cal_mean=True):
    means_prune, means_unprune = [], []
    prunes, unprunes = [], []
    for network in network_names:
        prune = df[(df["network"]==network) & (df["method"]=="map") & (df["prune"]==True) & (df["heuristic"]=="fill")]
        unprune = df[(df["network"]==network) & (df["method"]=="map") & (df["prune"]==False) & (df["heuristic"]=="fill")]
        if not cal_mean:
            prunes.append(prune["time"])
            unprunes.append(unprune["time"])
            continue
        mean_prune = round(prune["time"].mean(), 2)
        mean_unprune = round(unprune["time"].mean(), 2)
        means_prune.append(mean_prune)
        means_unprune.append(mean_unprune)
    if not cal_mean:
        return prunes, unprunes
    return means_prune, means_unprune

def compare_heuristic(df: pd.DataFrame, network_names, cal_mean=True):
    means_fill, means_degree = [], []
    fills, degrees = [], []
    for network in network_names:
        fill = df[(df["network"]==network) & (df["method"]=="map") & (df["prune"]==False) & (df["heuristic"]=="fill")]
        degree = df[(df["network"]==network) & (df["method"]=="map") & (df["prune"]==False) & (df["heuristic"]=="degree")]
        if not cal_mean:
            fills.append(fill["time"])
            degrees.append(degree["time"])
        mean_fill = round(fill["time"].mean(), 2)
        means_fill.append(mean_fill)
        mean_degree = round(degree["time"].mean(), 2)
        means_degree.append(mean_degree)
    if not cal_mean:
        return fills, degrees
    return mean_fill, mean_degree

def compare_qe_length(df: pd.DataFrame, network_names, cal_mean):
    qe_lens = []
    qe_means = []
    for network in network_names:
        new_df = df[(df["network"]==network) & (df["method"]=="mpe") & (df["prune"]==True) & (df["heuristic"]=="fill")]
        new_df["qe_length"] = new_df.apply(calculate_qe_len, axis=1)
        qe_len_set = set(new_df["qe_length"].to_list())
        qe_len_list = []
        mean_qe_len_list = []
        for qe_len in qe_len_set:
            qe_len_df = new_df[new_df["qe_length"]==qe_len]
            mean_qe_len = qe_len_df["time"].mean()
            qe_len_list.append(qe_len)
            if not cal_mean:
                mean_qe_len_list.append(qe_len_df["time"])
            else:
                mean_qe_len_list.append(mean_qe_len)
        qe_lens.append(qe_len_list)
        qe_means.append(mean_qe_len_list)
    return qe_lens, qe_means

def calculate_qe_len(x):
    q_len = len(list(eval(x["query"])))
    e_len = len(eval(x["evidence"]))
    # return q_len + e_len
    return e_len

def plot_double_bar(networks, y1, y2, save_name, pair=None):
    width = 0.2
    x = np.arange(len(networks))
    plt.figure(figsize=(8,6))
    plt.bar(x=x, height=y1, width=width, label=pair[0])
    plt.bar(x=x+width, height=y2, width=width, label=pair[1])
    for x_value, y_value in zip(x, y1):
        plt.text(x=x_value-0.02, y=y_value, s=y_value)
    for x_value, y_value in zip(x, y2):
        plt.text(x=x_value+width-0.02, y=y_value, s=y_value)
    plt.rcParams["axes.unicode_minus"] = False
    plt.title(f"Inference time {pair[0]} vs {pair[1]}")
    x_labels = networks
    plt.xticks(x+width/2, x_labels) 
    plt.ylabel("average run time (s) / network")
    plt.legend() 
    plt.savefig(save_name)
    plt.show()

def plot_double_box(networks, y1, y2, save_name, pair=None):
    width = 0.2
    x = np.arange(len(networks))
    plt.figure(figsize=(8,8))
    plt.boxplot(x=y1, positions=x, widths=width, showfliers=False, patch_artist=True ,boxprops={"facecolor": "red"})
    plt.boxplot(x=y2, positions=x+width, widths=width, showfliers=False, patch_artist=True ,boxprops={"facecolor": "blue"})
    plt.rcParams["axes.unicode_minus"] = False
    plt.title(f"Inference time {pair[0]} vs {pair[1]}")
    x_labels = networks
    plt.xticks(x+width/2, x_labels) 
    plt.ylabel("run time (s) / network")
    plt.legend() 
    plt.savefig(save_name)
    plt.show()

def plot_single_box(networks, x, y, save_name,):
    width = 0.2
    plt.figure(figsize=(16,14))
    for i, network in enumerate(networks):
        plt.subplot(2,3,i+1)
        plt.boxplot(x=y[i], positions=x[i], widths=width, showfliers=False, patch_artist=True ,boxprops={"facecolor": "red"})
        plt.xlabel("Evidence length")
        plt.title(network)
        plt.ylabel("Inference time (s)")


    plt.rcParams["axes.unicode_minus"] = False
    # plt.xlabel("(Q,e) sample length")
    # plt.ylabel("run time (s) / (Q,e) sample")

    plt.legend() 
    plt.savefig(save_name)
    plt.show()


def plot_with_line(networks, y1, y2, save_name, pair=None):
    
    num_sample = np.array(y1).shape[1]
    plt.figure(figsize=(16,12))
    for i, network in enumerate(networks):
        plt.subplot(2,3,i+1)

        plt.plot(np.arange(num_sample), y1[i], ls="-", c='r', alpha=0.5, label=pair[0])
        plt.plot(np.arange(num_sample), y2[i], ls="-", c='b', alpha=0.5, label=pair[1])
        plt.xlabel("Query & Evidence sample")


        plt.title(network)
        plt.ylabel("Inference time (s)")

    plt.legend()
    plt.savefig(save_name)
    plt.show()
    



file = "experiments/network10to50.csv"
# file = "experiments/network-30.csv"
df = load_data(file)
# compare_prune(df)
# compare_heuristic(df)
# compare_qe_length(df)

network_names = ["network-10", "network-20", "network-30", "network-40", "network-50"]

"""prune vs unprunes bar"""
means_prune, means_unprune = compare_prune(df, network_names)
plot_double_bar(network_names, means_prune, means_unprune, "prune_cps_bar.png",pair=["Prune", "UnPrune"])

# """fill vs degree bar"""
# means_fill, means_degree = compare_heuristic(df, network_names)
# plot_double_bar(network_names, means_fill, means_degree, pair=["min-fill", "min-degree"])

# """prune vs unprunes box"""
# prunes, unprunes = compare_prune(df, network_names, cal_mean=False)
# plot_double_box(network_names, prunes, unprunes, pair=["Prune", "UnPrune"])

# """fill vs degree box"""
# fills, degrees = compare_heuristic(df, network_names, cal_mean=False)
# plot_double_box(network_names, fills, degrees, "heuristic_cps_box", pair=["min-fill", "min-degree"])

"""prune vs unprunes plot"""
prunes, unprunes = compare_prune(df, network_names, cal_mean=False)
plot_with_line(network_names, prunes, unprunes, "prune_cps_line.png",pair=["Prune", "UnPrune"])

# """fill vs degree plot"""
# fills, degrees = compare_heuristic(df, network_names, cal_mean=False)
# plot_with_line(network_names, fills, degrees, "heuristic_cps_plot",pair=["min-fill", "min-degree"])

"""qe_len comparison plot"""
qe_lens, qe_means = compare_qe_length(df, network_names, cal_mean=False)
plot_single_box(network_names, qe_lens, qe_means, save_name="qe_lens_cpr.png")