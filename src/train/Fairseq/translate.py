#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator
from collections import defaultdict

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts
import logging

def main(opt):
    translator = build_translator(opt, report_score=True)
    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug)

    correct = 0
    total = 0
    per_length = defaultdict(lambda : {"correct" : 0, "total" : 0})
    trace = []
    with open(opt.src) as f_src, open(opt.tgt) as f_tgt, open(opt.output) as f_prd:
        for src, tgt, prd in zip(f_src, f_tgt, f_prd):
            if tgt == prd:
                correct += 1
            total += 1
            trace.append((src.strip(), tgt.strip(), prd.strip()))
    logging.info("Sequence Accuracy: {}".format(correct / total))

    with open(opt.output, 'w') as f_prd:
        f_prd.write("source\ttarget\tprediction\n")
        for src, tgt, prd in trace:
            f_prd.write(f"{src}\t{tgt}\t{prd}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
