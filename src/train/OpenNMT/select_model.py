"""
Python script acting on output traces provided by OpenNMT during model training.
Based on the validation accuracy, the script finds the best-performing
model, renames it and removes the other intermediate models saved.

OpenNMT version november 2018
"""
import os
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', type=str,
        help="Model name provided to OpenNMT, without the addition of the `_step_XXX.pt` affix."
    )
    parser.add_argument(
        '--steps', type=int,
        help="Integer value indicating the step size used in OpenNMT."
    )
    parser.add_argument(
        '--trace', type=str,
        help="Filename containing the OpenNMT output trace of training."
    )
    parser.add_argument(
        '--folder', type=str,
        help="Folder name containing the models."
    )
    args = vars(parser.parse_args())

    val_accs = []
    with open(args["trace"]) as file:
        for line in file:
            if not "Validation accuracy" in line:
                continue
            # Extract the validation accuracy of the current model
            performance = float(line.split()[-1])
            val_accs.append(performance)

    # Find the best-performing model out of the validation accuracies
    iterations = np.argmax(val_accs) * args["steps"]
    best_model_name = f"{args['model_name']}_step_{iterations}.pt"
    # Rename it to match the desired model name, without the step size
    os.rename(best_model_name, f"{args['model_name']}.pt")

    # Remove all other models that OpenNMT saved
    for filename in os.listdir(args["folder"]):
        if f"{args['model_name']}_step_" in os.path.join(args["folder"], filename):
            os.remove(os.path.join(args["folder"], filename))