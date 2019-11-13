import seaborn as sns
import numpy
import matplotlib.pyplot as plt

def plot_average(input_filename, output_filename, keep_ylabels=False):
    functions = ["swap", "echo", "copy", "repeat", "shift", "reverse", "append", "prepend", "remove_first", "remove_second"]
    # Increase parameters when reading out the files with results per length
    results = numpy.zeros((3, 10, 19))
    run = -1
    function = -1
    with open(input_filename) as f:
        for line in f:
            if "Run" in line:
                run += 1
                function = -1
            if "Function" in line:
                function += 1
            if "Length" in line:
                line = line.split()
                l = line[1]
                seqacc = line[-1]
                # Argument sizes run from 2 to 20
                results[run, function, int(l) - 2] = float(seqacc)

    # Figure size
    if keep_ylabels:
        sns.set(rc={'figure.figsize':(6,3.5), "lines.linewidth": 3})
    else:
        sns.set(rc={'figure.figsize':(4,3.5), "lines.linewidth": 3})
    n_chars = range(2, 21)
    results = results.reshape((1, 30, 19))
    average_all = numpy.mean(results, 1)[0]
    std_all = numpy.std(results, 1)[0]

    # Mean
    ax = sns.lineplot(x=n_chars, y=average_all, color="blue", marker="^")
    # - STD
    ax.fill_between(n_chars, average_all - std_all, average_all, color="blue", alpha=0.2)
    # + STD
    ax.fill_between(n_chars, average_all + std_all, average_all, color="blue", alpha=0.2)
    plt.xticks(list(range(2, 21, 2)))
    plt.ylim(-0.02, 1.02)
    plt.xlim(2, 20)
    ax.grid(False)

    # Share axes for subplots displayed next to each other in the paper
    if keep_ylabels:
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylabel("accuracy")
        ax.set_ylabel("accuracy",fontsize=20)
        ax.tick_params(labelsize=20)
    else:
        plt.yticks([])

    ax.set_xlabel("number of characters", fontsize=20)
    ax.tick_params(labelsize=20)
    plt.savefig(output_filename, bbox_inches='tight')
    plt.clf()


def plot_per_function(input_filename, output_filename, keep_ylabels=False, convs2s=False):
    functions = ["swap", "echo", "copy", "repeat", "shift", "reverse", "append", "prepend", "remove_first", "remove_second"]
    # Increase parameters when reading out the files with results per length
    results = numpy.zeros((3, 10, 19))
    run = -1
    function = -1
    with open(input_filename) as f:
        for line in f:
            if "Run" in line:
                run += 1
                function = -1
            if "Function" in line:
                function += 1
            if "Length" in line:
                line = line.split()
                l = line[1]
                seqacc = line[-1]
                # Argument sizes run from 2 to 20
                results[run, function, int(l) - 2] = float(seqacc)

    # Figure size
    if keep_ylabels:
        sns.set(rc={'figure.figsize':(6,3.5)})
    else:
        sns.set(rc={'figure.figsize':(4,3.5)})

    n_chars = range(2, 21)
    blues = sns.color_palette("Blues", 15)
    for i, f in enumerate(functions):
        # Mean over runs
        average_all = numpy.mean(results[:, i, :], 0)
        ax = sns.lineplot(x=n_chars, y=average_all, color=blues[i+5], marker="^", ax=None if i == 0 else ax, markeredgecolor='none')

    plt.xticks(list(range(2, 21)))
    plt.ylim(-0.02, 1.02)
    plt.xlim(2, 15)

    if convs2s:
        plt.text(3.7, 0.2, "reverse", horizontalalignment='left', size='large')
        plt.text(8.5, 0.5, "append", horizontalalignment='left', size='large')

    # Share axes for subplots displayed next to each other in the paper, but keep grid
    ax.grid(True)
    if keep_ylabels:
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylabel("accuracy")
        ax.set_ylabel("accuracy",fontsize=20)
        ax.tick_params(labelsize=20)
    else:
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_ticklabels([])

    ax.set_xlabel("number of characters",fontsize=20)
    ax.tick_params(labelsize=20)
    plt.savefig(output_filename, bbox_inches='tight')
    plt.clf()



if __name__ == "__main__":
    plot_per_function("localism2_lstms2s.txt", "lstms2s.pdf", keep_ylabels=True)
    plot_per_function("localism2_convs2s.txt", "convs2s.pdf", keep_ylabels=False, convs2s=True)
    plot_per_function("localism2_transformer.txt", "transformer.pdf", keep_ylabels=False)