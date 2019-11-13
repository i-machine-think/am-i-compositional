import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict


def depth(seq):
    if type(seq) is str:
        seq = seq.split()
    seq.append("END")
    queue = []
    new_seq = []
    open_brackets, depth = 0, 0

    for token in seq:
        if token in ["append", "prepend", "remove_first", "remove_second"]:
            new_seq.append(token)
            new_seq.append("(")
            open_brackets += 1
            queue.append(["two-place", 0])
        elif token in ["copy", "reverse", "shift", "echo", "swap_first_last", "repeat"]:
            new_seq.append(token)
            new_seq.append("(")
            open_brackets += 1
            queue.append(["one-place", 0])
        elif token == "," or token == "END":
            while len(queue) > 0:
                if queue[-1][0] == "one-place":
                    _ = queue.pop()
                    new_seq.append(")")
                    open_brackets -= 1
                elif queue[-1][0] == "two-place" and queue[-1][1] == 0:
                    queue[-1][1] = 1
                    break
                elif queue[-1][0] == "two-place" and queue[-1][1] == 1:
                    new_seq.append(")")
                    open_brackets -= 1
                    _ = queue.pop()
            if token == ",":
                new_seq.append(token)
        else:
            new_seq.append(token)
        if open_brackets > depth:
            depth = open_brackets
    assert new_seq.count("(") == new_seq.count(")"), \
        "Number of opening and closing brackets do not match."
    return(depth)


def count_functions(seq):
    functions = ["swap_first_last", "echo", "copy", "repeat", "shift",
                 "reverse", "append", "prepend", "remove_first", "remove_second"]
    n = 0
    seq = seq.split()
    for w in seq:
        if w in functions:
            n += 1
    return n


def load_results(prediction_files, source_file, target_file, mode):
    results = defaultdict(list)
    for prediction_file in prediction_files:
        with open(prediction_file) as f_pred, open(source_file) as f_src, \
                open(target_file) as f_target:
            for prediction, source, target in zip(f_pred, f_src, f_target):
                prediction = prediction.strip()
                source = source.strip()
                if mode == "depth":
                    index = depth(source)
                elif mode == "length":
                    index = len(source.split())
                elif mode == "number of functions":
                    index = count_functions(source)
                target = target.strip()
                results[index].append(int(prediction == target))
    x_ticks = sorted(results.keys())
    return x_ticks, [np.mean(results[i]) for i in x_ticks]


def plot(lstms2s, transformer, convs2s, source_file_opennmt, target_file_opennmt,
         source_file_fairseq, target_file_fairseq,
         output_filename, keep_ylabels=False, mode="depth"):
    x_lstms2s, y_lstms2s = load_results(lstms2s, source_file_opennmt, target_file_opennmt, mode)
    x_transf, y_transf = load_results(transformer, source_file_opennmt, target_file_opennmt, mode)
    x_convs2s, y_convs2s = load_results(convs2s, source_file_fairseq, target_file_fairseq, mode)

    # Figure size
    if keep_ylabels:
        sns.set(rc={'figure.figsize': (6, 3.5), "lines.linewidth": 3})
    else:
        sns.set(rc={'figure.figsize': (6, 3.5), "lines.linewidth": 3})

    blues = list(sns.color_palette("Blues", 10))
    ax = sns.lineplot(x=x_lstms2s, y=y_lstms2s, color=blues[3], marker=".",
                      label="LSTMS2S", markeredgecolor='none')
    ax.lines[0].set_linestyle("--")
    ax = sns.lineplot(x=x_convs2s, y=y_convs2s, color=blues[6], marker=".",
                      ax=ax, label="ConvS2S", markeredgecolor='none')
    ax.lines[1].set_linestyle(":")
    ax = sns.lineplot(x=x_transf, y=y_transf, color=blues[9], marker=".",
                      ax=ax, label="Transformer", markeredgecolor='none')

    if mode == "depth":
        plt.xlim(min(x_lstms2s), 14)
        plt.xticks(list(range(min(x_lstms2s), 14 + 1, 1)))
    elif mode == "number of functions":
        plt.xlim(min(x_lstms2s), 15)
        plt.xticks(list(range(min(x_lstms2s), 15 + 1, 1)))
    else:
        plt.xticks(list(range(5, 50 + 1, 5)))
        plt.xlim(min(x_lstms2s), 50)
        print(max(x_lstms2s))
    plt.ylim(-0.01, 1.01)

    # Share axes for subplots displayed next to each other in the paper, but keep grid
    ax.grid(True)
    ax.set_xlabel(mode, fontsize=20)
    ax.tick_params(labelsize=20)
    if keep_ylabels:
        plt.yticks(list([x / 10 for x in range(0, 12, 2)]))
        ax.set_ylabel("accuracy", fontsize=20)
        plt.legend(fontsize=15)
    else:
        plt.yticks(list([x / 10 for x in range(0, 12, 2)]))
        ax.yaxis.set_ticklabels([])
        ax.get_legend().set_visible(False)

    ax.tick_params(labelsize=20)
    plt.savefig(output_filename, bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    lstms2s = ["lstms2s_pred_1.txt", "lstms2s_pred_2.txt", "lstms2s_pred_3.txt"]
    transformer = ["transformer_pred_3.txt", "transformer_pred_2.txt", "transformer_pred_1.txt"]
    convs2s = ["convs2s_pred_1.txt", "convs2s_pred_2.txt", "convs2s_pred_3.txt"]
    plot(lstms2s, transformer, convs2s, "test_opennmt.src", "test_opennmt.tgt",
         "test_fairseq.src", "test_fairseq.tgt", "depth.pdf",
         keep_ylabels=True, mode="depth")
    plot(lstms2s, transformer, convs2s, "test_opennmt.src", "test_opennmt.tgt",
         "test_fairseq.src", "test_fairseq.tgt", "length.pdf",
         keep_ylabels=False, mode="length")
    plot(lstms2s, transformer, convs2s, "test_opennmt.src", "test_opennmt.tgt",
         "test_fairseq.src", "test_fairseq.tgt", "functions.pdf",
         keep_ylabels=False, mode="number of functions")
