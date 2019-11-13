import csv
import argparse
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_per_function(traces, output_filename, remove_y=False):
    """
    Create bar plot with the sequence accuracy per function, stds included.

    Args:
        traces: list of filenames containing the pooled seq acss per function
        output_filename: name for the pdf with the plots saved
        remove_y: remove functions names at y axis for pretty paper plots
    """
    functions = ["remove_second", "remove_first", "copy", "append", "echo", "prepend", "shift", "swap", "reverse", "repeat"]
    functions = list(reversed(functions))
    results_per_function = defaultdict(list)

    # Traces consist of tab separated listings with name and accuracy
    for trace in traces:
        trace = csv.DictReader(open(trace), delimiter="\t")
        for item in trace:
            # Rename "swap first last" to "swap"
            if item["function"] != "swap_first_last":
                function_name = item["function"]
            else:
                function_name = "swap"
            results_per_function[function_name].append(float(item["accuracy"]))

    sns.set(rc={"figure.figsize": (6, 4)})

    blues = list(sns.color_palette("Blues", 15))[3:]
    average = [np.mean(results_per_function[f]) for f in functions]
    std = [np.std(results_per_function[f]) for f in functions]
    ax = sns.barplot(x=average, y=functions, xerr=std, palette=blues,
                     orient="h", errwidth=0.2)

    # Label axes with names and x / yticks
    plt.xlim(0.5, 1)
    plt.ylim(-0.5, 9.5)
    ax.grid(True)
    plt.xlabel("accuracy")
    ax.set_xlabel("accuracy", fontsize=20)
    ax.tick_params(labelsize=20)

    if not remove_y:
        plt.yticks(list(range(0, 10)))
    else:
        ax.yaxis.set_ticklabels([])

    plt.xticks([0.6, 0.7, 0.8, 0.9, 1])
    ax.tick_params(labelsize=20)

    # Save figure to pdf
    plt.savefig(output_filename, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=str, nargs="+",
                        help="Filenames of pooled accuracies per function.")
    parser.add_argument("--remove_y", action="store_true",
                        help="Add flag if the Y-axis labels are to be removed.")
    parser.add_argument("--output_filename", type=str, default="plot.pdf",
                        help="Name of the pdf that is saved with the plots.")
    args = vars(parser.parse_args())

    plot_per_function(args["traces"], args["output_filename"],
                      remove_y=args["remove_y"])
