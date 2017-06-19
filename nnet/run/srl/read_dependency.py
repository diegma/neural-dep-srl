import numpy as np
import sys



# conll2009 labels
_DEP_LABELS = ['ROOT', 'ADV', 'ADV-GAP', 'AMOD', 'APPO', 'BNF', 'CONJ', 'COORD', 'DEP',
               'DEP-GAP', 'DIR', 'DIR-GAP', 'DIR-OPRD', 'DIR-PRD', 'DTV', 'EXT',
               'EXT-GAP', 'EXTR', 'GAP-LGS', 'GAP-LOC', 'GAP-LOC-PRD', 'GAP-MNR',
               'GAP-NMOD', 'GAP-OBJ', 'GAP-OPRD', 'GAP-PMOD', 'GAP-PRD', 'GAP-PRP',
               'GAP-SBJ', 'GAP-TMP', 'GAP-VC', 'HMOD', 'HYPH', 'IM', 'LGS', 'LOC',
               'LOC-OPRD', 'LOC-PRD', 'LOC-TMP', 'MNR', 'MNR-PRD', 'MNR-TMP', 'NAME',
               'NMOD', 'OBJ', 'OPRD', 'P', 'PMOD', 'POSTHON', 'PRD', 'PRD-PRP',
               'PRD-TMP', 'PRN', 'PRP', 'PRT', 'PUT', 'SBJ', 'SUB', 'SUFFIX',
               'TITLE', 'TMP', 'VC', 'VOC']



_DEP_LABELS = {label: i for i, label in enumerate(_DEP_LABELS)}

_N_LABELS = len(_DEP_LABELS)


def get_adj(deps, degree):
    batch_size = len(deps)
    _MAX_BATCH_LEN = 0

    for de in deps:
        if len(de) > _MAX_BATCH_LEN:
            _MAX_BATCH_LEN = len(de)

    _MAX_DEGREE = max(degree)

    adj_arc_in = np.zeros((batch_size * _MAX_BATCH_LEN, 2), dtype='int32')
    adj_lab_in = np.zeros((batch_size * _MAX_BATCH_LEN, 1), dtype='int32')
    adj_arc_out = np.zeros((batch_size * _MAX_BATCH_LEN*_MAX_DEGREE, 2), dtype='int32')
    adj_lab_out = np.zeros((batch_size * _MAX_BATCH_LEN*_MAX_DEGREE, 1), dtype='int32')


    mask_in = np.zeros((batch_size * _MAX_BATCH_LEN), dtype='float32')
    mask_out = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE), dtype='float32')

    mask_loop = np.ones((batch_size * _MAX_BATCH_LEN, 1), dtype='float32')

    tmp_in = {}
    tmp_out = {}

    for d, de in enumerate(deps):
        for a, arc in enumerate(de):
            if arc[0] != 'ROOT' and arc[0] in _DEP_LABELS:
                arc_1 = int(arc[1])-1
                arc_2 = int(arc[2])-1
                if a in tmp_in:
                    tmp_in[a] += 1
                else:
                    tmp_in[a] = 0

                if arc_2 in tmp_out:
                    tmp_out[arc_2] += 1
                else:
                    tmp_out[arc_2] = 0

                idx_in = (d * _MAX_BATCH_LEN) + a + tmp_in[a]
                idx_out = (d * _MAX_BATCH_LEN * _MAX_DEGREE) + arc_2 * _MAX_DEGREE + tmp_out[arc_2]

                adj_arc_in[idx_in] = np.array([d, arc_2])  # incoming arcs
                adj_lab_in[idx_in] = np.array([_DEP_LABELS[arc[0]]])  # incoming arcs

                mask_in[idx_in] = 1.

                if tmp_out[arc_2] < _MAX_DEGREE:
                    adj_arc_out[idx_out] = np.array([d, arc_1])  # outgoing arcs
                    adj_lab_out[idx_out] = np.array([_DEP_LABELS[arc[0]]])  # outgoing arcs
                    mask_out[idx_out] = 1.

        tmp_in = {}
        tmp_out = {}

    return np.transpose(adj_arc_in), np.transpose(adj_arc_out), \
           np.transpose(adj_lab_in), np.transpose(adj_lab_out), \
           mask_in.reshape((len(deps) * _MAX_BATCH_LEN, 1)), \
           mask_out.reshape((len(deps) * _MAX_BATCH_LEN, _MAX_DEGREE)), \
           mask_loop
