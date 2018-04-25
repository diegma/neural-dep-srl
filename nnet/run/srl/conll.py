import argparse
import itertools
import json
import re
import sys

from collections import defaultdict


def read_sentences(data):
    for key, group in itertools.groupby(data, lambda x: not x.strip()):
        if not key:
            yield list(group)


def process_frame_2009(frame_ids, arguments, frames):
    arguments = list(zip(*arguments))
    assert (len(arguments) == len(frame_ids))

    frame_data = defaultdict(list)
    for frame_id, arg, targ in zip(frame_ids, arguments, frames):
        roles = [label if label != '_' else 'O' for label in arg]
        frm = [l if l != '_' else 'O' for l in targ]

        for i, x in enumerate(frm):
            if x != 'O':
                idx = i

        frame_data['f_' + frame_id].append({
            'roles': roles,
            'frames': frm,
            'target': {
                'index': [list(range(idx, idx + 1))]
            }
        })

    return frame_data


def arg_parse():
    parser = argparse.ArgumentParser("SRL argument extractor")
    parser.add_argument("--data", required=True)
    parser.add_argument("--preserve-sense", action='store_true')

    return parser.parse_args()


dict_labels = dict()


def from_2009(block, preserve_sense):
    record = defaultdict(list)
    block_frames, block_arguments, block_targets = [], [], []
    for line in block:
        parts = re.split("\t", line.strip())
        d_head = parts[0]
        d_tail = parts[9]
        d_label = parts[11]
        word = parts[1]
        pos_tag = parts[5]
        if parts[13] != '_':
            if preserve_sense:
                predicate = parts[13]
            else:
                predicate, frame_set_id = parts[13].split('.')
            block_targets.append(['_' for i in block])
            block_targets[-1][int(parts[0]) - 1] = predicate
        else:
            predicate = '_'
            frame_set_id = '_'
        record['tokenized_sentence'].append(word)
        record['pos'].append(pos_tag)

        record['d_parsing'].append([
            d_label, [int(d_tail), '_'], [int(d_head), word]
        ])

        arguments = parts[14:]
        for a in arguments:
            if a not in dict_labels:
                dict_labels[a] = 1

        if predicate != '_':
            block_frames.append(predicate)

        block_arguments.append(arguments)

    frame_data = process_frame_2009(block_frames, block_arguments, block_targets)
    record.update(frame_data)
    return record


def main():
    a = arg_parse()
    data = open(a.data)
    corpus = defaultdict(dict)
    for doc_id, block in enumerate(read_sentences(data)):
        corpus[doc_id][doc_id] = from_2009(block, a.preserve_sense)
    print(json.dumps(corpus, indent=4))

if __name__ == '__main__':
    main()
