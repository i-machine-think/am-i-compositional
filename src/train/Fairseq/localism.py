"""
Adapted Fairseq toolkit translation script, that computes the localism score
of the "unrolled" versions of train and test datasets.

TODO: cleanup, comment, specify data formats
"""

from collections import namedtuple, defaultdict
from tqdm import tqdm
import random
import numpy as np
import torch
import logging

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator



def make_batches(lines, args, task, max_positions):
    tokens = [
        tokenizer.Tokenizer.tokenize(src_str, task.source_dictionary, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = np.array([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=data.LanguagePairDataset(tokens, lengths, task.source_dictionary),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            srcs=[lines[i] for i in batch['id']],
            tokens=batch['net_input']['src_tokens'],
            lengths=batch['net_input']['src_lengths'],
        ), batch['id']


def main(args):
    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(':')
    models, model_args = utils.load_ensemble_for_inference(
        model_paths, task, model_arg_overrides=eval(args.model_overrides)
    )

    # Set dictionaries
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    # Initialize generator
    translator = SequenceGenerator(
        models, tgt_dict,
        beam_size=args.beam,
        minlen=args.min_len,
        stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized),
        len_penalty=args.lenpen,
        unk_penalty=args.unkpen,
        sampling=args.sampling,
        sampling_topk=args.sampling_topk,
        sampling_temperature=args.sampling_temperature,
        diverse_beam_groups=args.diverse_beam_groups,
        diverse_beam_strength=args.diverse_beam_strength,
    )

    if use_cuda:
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    def make_result(src_str, hypos):
        result = Translation(
            src_str='O\t{}'.format(src_str),
            hypos=[],
            pos_scores=[],
            alignments=[],
        )

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            result.pos_scores.append('P\t{}'.format(
                ' '.join(map(
                    lambda x: '{:.4f}'.format(x),
                    hypo['positional_scores'].tolist(),
                ))
            ))
            result.alignments.append(
                'A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment)))
                if args.print_alignment else None
            )
        return result

    def process_batch(batch):
        tokens = batch.tokens
        lengths = batch.lengths

        if use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        encoder_input = {'src_tokens': tokens, 'src_lengths': lengths}
        translations = translator.generate(
            encoder_input,
            maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
        )

        return [make_result(batch.srcs[i], t) for i, t in enumerate(translations)]


    def translate_one(source, args, task, max_positions):
        for batch, batch_indices in make_batches([source], args, task, max_positions):
            result = process_batch(batch)[0]
            for hypo, pos_scores, align in zip(result.hypos, result.pos_scores, result.alignments):
                hypo = hypo.split("\t")[-1]         
                break
        return hypo

    def process_unrolled(steps, args, task, max_positions):
        is_long = False
        pairs = []
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
            predicted_target = translate_one(source, args, task, max_positions)
            pairs.append((source, predicted_target))

            if "*" in target[0]:
                collect_outcomes[target[0]] = predicted_target
            else:
                return predicted_target, pairs
        logging.error("Unrolling stopped prematurely.")


    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )
    print(translate_one("copy G17 P19 Z18 E1 S13 J15 A3 A3", args, task, max_positions))


    data = []
    with open(args.src) as f:
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
    scores_per_input_length = defaultdict(list)
    scores_per_target_length = defaultdict(list)
    all_pairs = []

    random.shuffle(data)

    for sample in tqdm(data[:1000]):
        unrolled_predicted, is_long, pairs = process_unrolled(
            sample["unrolled"], args, task, max_positions
        )
        source, target = sample["original"]
        target = " ".join(target)
        source = " ".join(source)
        original_predicted = translate_one(source, args, task, max_positions)
        local_score = original_predicted == unrolled_predicted

        all_pairs.append((pairs, local_score))
        predictions_equal.append(local_score)
        scores_per_input_length[len(source)].append(local_score)
        scores_per_target_length[len(target)].append(local_score)

    print(f"Localism {np.mean(predictions_equal)}")

    with open("trace_localism.txt", 'w') as f:
        for pairs, score in all_pairs:
            f.write("------------------------------\n")
            for s, t in pairs:
                f.write("{} -> {}\n".format(s, t))
            f.write("{}".format(score))

    # # UNCOMMENT TO COMPUTE LOCALISM SCORE PER INPUT / TARGET LENGTH
    # for key in sorted(list(scores_per_input_length.keys())):
    #     scores = scores_per_input_length[key]
    #     print(f"Input length {key}, {len(scores)}, localism score {sum(scores)/len(scores)}.")

    # for key in sorted(list(scores_per_target_length.keys())):
    #     scores = scores_per_target_length[key]
    #     print(f"Target length {key}, localism score {sum(scores) / len(scores)}.")


if __name__ == '__main__':
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)

    Batch = namedtuple('Batch', 'srcs tokens lengths')
    Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')

    main(args)
