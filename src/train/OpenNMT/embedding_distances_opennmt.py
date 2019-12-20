import torch
import argparse
import onmt
import onmt.model_builder
import onmt.inputters
import onmt.opts
from scipy import spatial
import numpy as np

from onmt.utils.logging import init_logger, logger
from collections import defaultdict


def compute_distances(model_name, opt):

    """1 - OpenNMT code to extract embeddings from trained models."""
    # Add in default model arguments, possibly added since training.
    checkpoint = torch.load(model_name,
                            map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']
    src_dict = None

    # the vocab object is a list of tuple (name, torchtext.Vocab)
    # we iterate over this list and associate vocabularies based on the name
    for vocab in checkpoint['vocab']:
        if vocab[0] == 'src':
            src_dict = vocab[1]
    assert src_dict is not None

    fields = onmt.inputters.load_fields_from_vocab(checkpoint['vocab'])
    model_opt = checkpoint['opt']
    for arg in dummy_opt.__dict__:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt.__dict__[arg]

    model = onmt.model_builder.build_base_model(
        model_opt, fields, False, checkpoint)
    embeddings = model.encoder.embeddings.word_lut.weight.data.tolist()

    """2 - Added src code by am-i-compositional team."""
    emb_dict = {}
    distance_per_token = defaultdict(lambda: {"twin": 0, "other": []})

    for i, embedding in enumerate(embeddings):
        emb_dict[src_dict.itos[i]] = embedding

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
    init_logger('extract_embeddings.log')
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model', required=True, type=str, nargs="+",
                        help='Path to model .pt file')
    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    opt = parser.parse_args()

    averaged_over_models = defaultdict(lambda : {"twin": [], "other": []})

    # Compute distance statistics per model
    for i, model_name in enumerate(opt.model):
        print(f"Model {i+1}\n-------------------------------")
        averaged = compute_distances(model_name, opt)
        for token in averaged:
            averaged_over_models[token]["twin"].append(averaged[token][0])
            averaged_over_models[token]["other"].append(averaged[token][1])
        print("\n")

    # Average across models
    total_twin = 0
    total_other = 0

    print(f"Averaged {i+1}\n-------------------------------")
    for token in averaged_over_models:
        distance_twin = np.mean(averaged_over_models[token]['twin'])
        total_twin += distance_twin
        distance_twin_std = np.std(averaged_over_models[token]['twin'])
        distance_other = np.mean(averaged_over_models[token]['other'])
        total_other += distance_other
        distance_other_std = np.std(averaged_over_models[token]['other'])

        print(f"{token} vs twin: {distance_twin:.3f}+/-{distance_twin_std:.3f}, " +
              f"{token} vs other: {distance_other:.3f}+/-{distance_other_std:.3f}")

    print(f"avg vs twin: {total_twin/4}, " +
          f"avg vs other: {total_other/4}.")
