import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd

def interpret(lines):
    accuracies = []
    for line in lines:
        line = line.strip()
        # Catch the cases for which convs2s finished early
        if "Sequence Accuracy" in line and not "ERROR" in line:
            accuracies.append(float(line.split()[-1]))
    adapted, original = [], []
    for i, x in enumerate(accuracies):
        # Adapted and original accuracies alternate
        if i % 2 == 0:
            adapted.append(x)
        else:
            original.append(x)
    # Pad for missing accuracies due to early convergence
    a = [0] + adapted[:25]
    a = a + (28 - len(a)) * [a[-1]]
    b = [0] + original[:25]
    b = b + (28 - len(b)) * [b[-1]]
    return np.array(a), np.array(b)


def plot_model(model, ratio, keep_x_axis, keep_y_axis):
    adapted, original = [], []

    # Collect overall accuracies
    filename = f"{model}-results/{model}_evaluate_ratio={ratio}_run="
    adapted1, original1 = interpret(open(filename + "1.txt").readlines())
    adapted2, original2 = interpret(open(filename + "2.txt").readlines())
    adapted3, original3 = interpret(open(filename + "3.txt").readlines())
    # Collect the average trends and standard deviations
    adapted_average = np.mean(np.array([adapted1, adapted2, adapted3]), axis=0)
    # Running average
    adapted_average = np.convolve(adapted_average, np.ones((3,))/3, mode='valid')[:26]
    adapted.append(adapted_average)
    adapted_average = [1 - x for x in adapted_average]
    adapted_std = np.std(np.array([adapted1, adapted2, adapted3]), axis=0)[:26]
    original_average = np.mean(np.array([original1, original2, original3]), axis=0)
    # Running average
    original_average = np.convolve(original_average, np.ones((3,))/3, mode='valid')[:26]
    original_std = np.std(np.array([original1, original2, original3]), axis=0)[:26]
    original.append(original_average)
    steps = range(0, len(original_average))
    top = [1 for x in steps]
    sns.set()

    # Set the style
    sns.set_context('paper')
    sns.set_style("whitegrid")
    sns.set_context(rc={"lines.linewidth": 1.5})
    f = plt.figure(figsize=(6,3))
    sns.set(font_scale=1.75)
    plt.ylim(0, 1)
    plt.xlim(0, 25)

    # Create overgeneralisation in red, memorisation in blue
    ax = sns.lineplot(steps, adapted_average, color='blue')
    ax = sns.lineplot(steps, original_average, color='red', legend=False, ax=ax)

    # Fill up the areas
    ax.fill_between(steps, original_average, adapted_average, alpha=0.2, color='grey')
    ax.fill_between(steps, adapted_average, top, alpha=0.12, color='blue')
    ax.fill_between(steps, original_average, alpha=0.2, color='red')

    # Fill standard deviations 
    ax.fill_between(steps, original_average - original_std,
                    original_average + original_std, alpha=0.2, color='red')
    ax.fill_between(steps, adapted_average - adapted_std,
                    adapted_average + adapted_std, alpha=0.2, color='blue')
    ax.grid(False)
    
    # Create the axes where needed
    if keep_x_axis: plt.xlabel("epoch")
    else: ax.set_xticklabels([])
    if keep_y_axis: plt.ylabel("accuracy")
    else: ax.set_yticklabels([])

    # Add overgeneralisation and memorisation labels
    if max(original_average) > 0.5:
        plt.text(0.5, 0.07, "overgeneralisation", horizontalalignment='left', size='medium', color='black')
    if min(adapted_average) < 0.5:
        plt.text(15, 0.8, "memorisation", horizontalalignment='left', size='medium', color='black')
    f.savefig(f"{model}ratio={str(ratio * 100).replace('.', '')}.pdf", bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    plot_model("lstms2s", 0.0001, False, True)
    plot_model("lstms2s", 0.0005, False, True)
    plot_model("lstms2s", 0.001, False, True)
    plot_model("lstms2s", 0.005, True, True)
    plot_model("convs2s", 0.0001, False, False)
    plot_model("convs2s", 0.0005, False, False)
    plot_model("convs2s", 0.001, False, False)
    plot_model("convs2s", 0.005, True, False)
    plot_model("transformer", 0.0001, False, False)
    plot_model("transformer", 0.0005, False, False)
    plot_model("transformer", 0.001, False, False)
    plot_model("transformer", 0.005, True, False)