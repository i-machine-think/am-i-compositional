"""
Script to distances between embeddings for the am-i-compositional
substitutivity experiments.
"""
from collections import defaultdict
import numpy as np
from scipy import spatial
from fairseq import options, tasks, utils


def main(args):
    """1 - Load Fairseq models in conventional Fairseq way."""
    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    model_paths = args.path.split(':')
    models, _ = utils.load_ensemble_for_inference(
        model_paths, task, model_arg_overrides=eval(args.model_overrides)
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    model = models[0]

    """2 - Added src code by am-i-compositional team."""
    emb_dict = {}
    distance_per_token = defaultdict(lambda: {"twin": 0, "other": []})
    for i, embedding in enumerate(model.encoder.embed_tokens.weights):
        emb_dict[src_dict.symbols[i]] = embedding

    # For every adapted function, compute distance to its twin,
    # and to all other functions
    for f in ["repeat", "remove_second", "swap_first_last", "append"]:
        functions = ["repeat", "swap_first_last", "reverse", "echo", "copy",
                     "shift", "append", "prepend", "remove_first",
                     "remove_second"]
        functions.remove(f)

        for f2 in functions:
            distance = spatial.distance.cosine(emb_dict[f], emb_dict[f2])
            distance_per_token[f]["other"].append(distance)

        twin = f + "_twin"
        distance = spatial.distance.cosine(emb_dict[f], emb_dict[twin])
        distance_per_token[f]["twin"] = distance

    averaged = dict()
    for token in distance_per_token:
        distance_twin = distance_per_token[token]['twin']
        distance_other = np.mean(distance_per_token[token]['other'])
        print(f"{token} vs twin: {distance_twin:.3f}, " +
              f"{token} vs other: {distance_other:.3f}")
        averaged[token] = (distance_twin, distance_other)
    return averaged


if __name__ == "__main__":
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)
