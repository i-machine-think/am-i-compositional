#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
import random
import logging

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts

from tqdm import tqdm


from collections import defaultdict

def process_unrolled(steps, translator):
    pairs = []
    is_long = False
    collect_outcomes = dict()
    for source, target in steps:
        for i, token in enumerate(source):
            if "*" in token: source[i] = collect_outcomes[token]

        source2 = []
        for token in source:
            if type(token) != list:
                source2.append(token)
            else:
                source2.extend(token)
        source = source2

        source = " ".join(source)
        predicted_target = translator.translate_one(src_data_iter=[source], batch_size=1)
        pairs.append((source, " ".join(predicted_target)))

        if "<eos>" in predicted_target: predicted_target.remove("<eos>")

        if "*" in target[0]:
            collect_outcomes[target[0]] = predicted_target
            if len(predicted_target) > 5: is_long = True
        else:
            return predicted_target, is_long, pairs
    logging.error("Unrolled sequence stopped before there was a final predicted target.")


def main(opt):
    translator = build_translator(opt, report_score=True)

    data = []
    with open(opt.src) as f:
        sample = {"unrolled" : [], "original": ("", "")}
        for line in f:
            [line_type, source, target] = line.split("\t")
            source = source.strip().split()
            target = target.strip().split()
            if line_type == "unrolled":
                sample[line_type].append((source, target))
            else:
                sample[line_type] = (source, target)
                data.append(sample)
                sample = {"unrolled" : [], "original": ("", "")}

    predictions_equal = []
    performance = []
    scores_per_input_length = defaultdict(list)
    scores_per_target_length = defaultdict(list)
    lengths = []
    shorts = []
    length_is_score = []
    all_pairs = []

    random.shuffle(data)
    for sample in data[:10000]:
        unrolled_predicted, is_long, pairs = process_unrolled(sample["unrolled"], translator)
        source, target = sample["original"]
        source = " ".join(source)
        original_predicted = translator.translate_one(src_data_iter=[source], batch_size=1)
        if "<eos>" in original_predicted: original_predicted.remove("<eos>")
        if "<eos>" in unrolled_predicted: unrolled_predicted.remove("<eos>")
        local_score = original_predicted == unrolled_predicted
        all_pairs.append((pairs, local_score))

        if is_long:
            lengths.append(local_score)
        else:
            shorts.append(local_score)
        predictions_equal.append(local_score)
        length_is_score.append(local_score == (not is_long))
        performance.append(unrolled_predicted == target)
        scores_per_input_length[len(source)].append(original_predicted == unrolled_predicted)
        scores_per_target_length[len(target)].append(original_predicted == unrolled_predicted)

    logging.info("Localism {}, Performance {}".format(sum(predictions_equal) / len(predictions_equal), sum(performance) / len(performance)))
    logging.info("Score of sequences containing strings > 5: {}".format(sum(lengths) / len(lengths)))
    logging.info("Length equals score: {}".format(sum(length_is_score) / len(length_is_score)))
    logging.info("Percentage of long ones: {}".format(len(lengths) / len(predictions_equal)))
    logging.info("Percentage of short ones: {}".format(len(shorts) / len(predictions_equal)))
    logging.info("Score of sequences containing strings <= 5: {}".format(sum(shorts) / len(shorts)))

    #with open("trace_localism.txt", 'w') as f:
    #    for pairs, score in all_pairs:
    #        f.write("-----------------------------\n")
    #        for s, t in pairs:
    #            f.write("{} -> {}\n".format(s, t))
    #        f.write("{}\n".format(score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
