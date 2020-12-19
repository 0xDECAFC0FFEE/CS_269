import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from itertools import chain
import random

def load_log(path):
    if (path/"test_accs.txt").exists():
        with open(path/"test_accs.txt", "r") as handle:
            test_accs = json.load(handle)
        with open(path/"expr_params.json", "r") as handle:
            expr_params = json.load(handle)
        pr = expr_params["prune_strategy"]["rate"]
        sparsity = [1-(1-pr)**i for i in range(len(test_accs))]
        test_accs = [{"sparsity": s, "test_accs": acc, "prune_iter": i} for (s, acc, i) in zip(sparsity, test_accs, range(len(test_accs)))]
    else:
        with open(path/"test_accs.json", "r") as handle:
            test_accs = json.load(handle)
        test_accs = [{"sparsity":score["prune_rate"], "test_accs": score["test_accs"]} for score in test_accs]

    return test_accs

def sparsity_to_iter(s, pr):
    return abs(int(round(np.log(1-s)/np.log(1-pr))))

def iter_to_sparsity(i, pr):
    return 1-(1-pr)**i


def max_accs(log):
    early_stop_epoch = []
    early_stop_acc = []
    sparsities = []

    for prune_iter in log:
        early_stop_epoch.append(0)
        early_stop_acc.append(0)
        sparsities.append(prune_iter["sparsity"])
        accs = prune_iter["test_accs"]

        for epoch, score in enumerate(accs):
            if score > early_stop_acc[-1]:
                early_stop_epoch[-1] = epoch
                early_stop_acc[-1] = score

    return [{"sparsity": s, "epoch": e, "max_acc": a} for s, e, a in zip(sparsities, early_stop_epoch, early_stop_acc)]

def early_stopping_stats(log):
    early_stop_epoch = []
    early_stop_acc = []
    sparsities = []

    for prune_iter in log:
        early_stop_epoch.append(0)
        early_stop_acc.append(0)
        sparsities.append(prune_iter["sparsity"])
        accs = prune_iter["test_accs"]

        for epoch, score in enumerate(accs):
            if score > early_stop_acc[-1]:
                early_stop_epoch[-1] = epoch
                early_stop_acc[-1] = score
            elif score < early_stop_acc[-1]:
                break

    return [{"sparsity": s, "epoch": e, "acc": a} for s, e, a in zip(sparsities, early_stop_epoch, early_stop_acc)]

def plot_finetuning_accs(lth_accs):
    title = "lth maml prune iteration test finetuning accuracies"
    print(f'drawing graph for "{title}"')
    plt.figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

    for i, score in enumerate(lth_accs):
        sparsity = score["sparsity"]
        acc = score["test_accs"]
        plt.plot(acc, "o-", color=(0, (i+5)/(len(lth_accs)+5), 0), label=f"prune iter {i+1}, {round(sparsity*100)}% sparsity")

    plt.title(title)
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
    plt.xticks(ticks=range(len(acc)))
    plt.xlabel("epoch")
    plt.ylabel("finetuning test accuracy")

    plt.gcf().subplots_adjust(bottom=0.10)

    plt.savefig(graph_dir/f"{title.replace(' ', '_')}.png", dpi=200)
    plt.clf()

corder = [
    np.array([112, 143, 255])/255,
    np.array([205, 130, 255])/255,
    np.array([255, 166, 166])/255,
]

def get_random_color(opacity=1):
    if len(corder) > 0:
        r, g, b = corder.pop()
    else:
        r = random.randrange(0, 100)/100
        g = random.randrange(0, 100)/100
        b = random.randrange(0, 100)/100
    return [r, g, b, opacity]

def plot_max_accs(plot_accs, scatter_accs, sparsity_range=(0.0, .9)):
    title = "max test acc generally increases after pruning"
    print(f'drawing graph for "{title}"')
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    for name, max_accs in plot_accs:
        plt.plot([score["sparsity"] for score in max_accs], [score["max_acc"] for score in max_accs], label=name, c=get_random_color())

    for name, max_accs in scatter_accs:
        plt.scatter([score["sparsity"] for score in max_accs], [score["max_acc"] for score in max_accs], label=name, c=[get_random_color()])

    plt.title(title)
    plt.xlabel("sparsity")
    plt.ylabel("max test acc")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.savefig(graph_dir/f"{title.replace(' ', '_')}.png", dpi=200)
    plt.clf()

def plot_finetuning_time(plot_accs, scatter_accs, iterations=14, pr=.1, time_per_iter=6.4):
    title = "finetuning time on raspberry pi generally decreases after pruning"
    print(f'drawing graph for "{title}"')

    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    for name, scores in plot_accs:
        plt.plot([sparsity_to_iter(score["sparsity"], pr) for score in scores], [score["epoch"] for score in scores], label=name, c=get_random_color())

    for name, scores in scatter_accs:
        plt.scatter([sparsity_to_iter(score["sparsity"], pr) for score in scores], [score["epoch"] for score in scores], label=name, c=[get_random_color()])

    max_iter, max_epoch = 0, 0
    for name, scores in plot_accs + scatter_accs:
        for score in scores:
            iter = sparsity_to_iter(score["sparsity"], pr)
            epoch = score["epoch"]
            max_iter = max(max_iter, abs(int(iter)))
            max_epoch = max(max_epoch, abs(int(epoch)))

    plt.title(title)
    plt.xticks(ticks=range(1, max_iter+1), labels=[f"{round(iter_to_sparsity(i, pr)*100)}% / {i}" for i in range(1, max_iter+1)], rotation=-40)
    plt.xlabel("sparsity / prune iteration (lth maml only)")
    plt.ylabel("epoch / rpi2 finetuning time (lth maml only)")
    plt.yticks(ticks=range(1, max_epoch+1), labels=[f"{i} / {int(i*time_per_iter)} min" for i in range(1, max_epoch+1)])
    # plt.gcf().subplots_adjust(left=.19, bottom=0.2)
    plt.gcf().subplots_adjust(left=.15, bottom=0.15, right=.74)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.legend(bbox_to_anchor=(-.2, -.3), loc='lower left')
    plt.savefig(graph_dir/f"{title.replace(' ', '_')}.png", dpi=200)
    plt.clf()


graph_dir = Path("graphs")

log_files = [
    Path("logs")/"expr|2020-12-06|17:31:18|7CpKn|lth_maml", 
    Path("logs")/"expr|2020-12-14|19:51:11|rx3xi|lth_maml|cudnn_disabled|try_to_repeat_fB9nb|1ajhv_prune_rate_too_low"
]
maml_lth_accs = [load_log(log_file) for log_file in log_files]
maml_max_accs = [max_accs(log) for log in maml_lth_accs]
maml_early_stopping = [early_stopping_stats(log) for log in maml_lth_accs]

smaml_accs = [
    {"sparsity": .27, "test_accs": [0.1957, 0.4026, 0.4214, 0.4224, 0.4253, 0.4263, 0.4287, 0.4285, 0.4292, 0.429 , 0.43]}
]
smaml_max_acc = max_accs(smaml_accs)
smaml_early_stopping = early_stopping_stats(smaml_accs)

rigl_accs = [
    {"sparsity": .10, "test_accs": [0.2003, 0.4373, 0.438, 0.4404, 0.4424, 0.4421, 0.4426, 0.4426, 0.4438, 0.4446, 0.4443]},
    {"sparsity": .19, "test_accs": [0.2039, 0.4272, 0.437, 0.4402, 0.4421, 0.4434, 0.445, 0.4446, 0.445, 0.4443, 0.4446]},
    {"sparsity": .27, "test_accs": [0.19, 0.4175, 0.4275, 0.4294, 0.4324, 0.4324, 0.4324, 0.4329, 0.432, 0.433, 0.433]},
    {"sparsity": .34, "test_accs": [0.2019, 0.4343, 0.4404, 0.4417, 0.4417, 0.443, 0.4438, 0.4429, 0.4429, 0.4429, 0.4424]},
    {"sparsity": .41, "test_accs": [0.2013, 0.4355, 0.4463, 0.4482, 0.4485, 0.4495, 0.4492, 0.45, 0.45, 0.4502, 0.4512]},
    {"sparsity": .47, "test_accs": [0.19, 0.4377, 0.4426, 0.4443, 0.446, 0.4453, 0.4456, 0.4463, 0.446, 0.447, 0.447]},
    {"sparsity": .52, "test_accs": [0.1951, 0.4155, 0.4246, 0.427, 0.426, 0.4287, 0.4292, 0.4294, 0.4297, 0.43, 0.431]},
    {"sparsity": .57, "test_accs": [0.1974, 0.4348, 0.4397, 0.4453, 0.4458, 0.4465, 0.448, 0.4487, 0.4487, 0.4492, 0.4492]},
    {"sparsity": .61, "test_accs": [0.2036, 0.4482, 0.4568, 0.4578, 0.4583, 0.458, 0.4578, 0.4587, 0.4587, 0.4587, 0.4585]},
    {"sparsity": .65, "test_accs": [0.2078, 0.436, 0.4463, 0.4482, 0.4485, 0.45, 0.4487, 0.4485, 0.4487, 0.4502, 0.4497]},
    {"sparsity": .69, "test_accs": [0.1918, 0.4219, 0.4316, 0.4358, 0.4382, 0.4395, 0.4397, 0.441, 0.4412, 0.441, 0.4407]},
    {"sparsity": .72, "test_accs": [0.1993, 0.4397, 0.4502, 0.4521, 0.4531, 0.4534, 0.4534, 0.4526, 0.4517, 0.452, 0.452]},
    {"sparsity": .75, "test_accs": [0.187, 0.428, 0.44, 0.4424, 0.443, 0.4443, 0.4434, 0.4429, 0.443, 0.4426, 0.4417]},
    {"sparsity": .77, "test_accs": [0.2048, 0.4346, 0.4443, 0.4463, 0.4463, 0.4475, 0.449, 0.4492, 0.45, 0.4497, 0.4507]},
]
rigl_max_acc = max_accs(rigl_accs)
rigl_early_stopping = early_stopping_stats(rigl_accs)


print("loaded data; building graphs")

plot_finetuning_accs(maml_lth_accs[0])

plot_accs = [(f"lth maml run {i}", score) for i, score in enumerate(maml_max_accs)]
scatter_accs = [("rigl", rigl_max_acc), ("smaml", smaml_max_acc)]
plot_max_accs(plot_accs, scatter_accs)

plot_accs = [(f"lth maml run {i}", score) for i, score in enumerate(maml_max_accs)]
scatter_accs = [("rigl", rigl_max_acc), ("smaml", smaml_max_acc)]
# plot_finetuning_time(plot_accs, scatter_accs, pr=.1)