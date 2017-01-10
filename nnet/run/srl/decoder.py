import heapq
import math
import sys
from collections import namedtuple

State = namedtuple('State', ['score', 'label', 'prev', 'roles'])

_PRUNING_THRESHOLD = 5.


def _get_score(state):
    return state.score

def _continuation_constraint_no_bio(state):
    if not state.label.startswith('C-'):
        return True

    if state.label[2:] in state.roles:
        return True

    return False

CONLL2009_CONSTRAINTS = []


def constrained_decoder(voc, predictions, beam, constraints):
    heap = [State(score=0, label='O', prev=None, roles=set())]
    for i, prediction in enumerate(predictions):
        next_generation = list()
        for prev in heapq.nsmallest(beam, heap, key=_get_score):
            for j, prob in enumerate(prediction):
                label = voc[j]
                score = -math.log2(prob + sys.float_info.min)
                if score > _PRUNING_THRESHOLD and next_generation:
                    continue

                next_state = State(score=score + prev.score,
                                   label=label, prev=prev,
                                   roles=prev.roles)

                constraints_violated = [not check(next_state) for check in
                                        constraints]
                if any(constraints_violated):
                    continue

                next_generation.append(
                    State(next_state.score, next_state.label, next_state.prev,
                          next_state.roles | {next_state.label[2:]}))

        heap = next_generation

    head = heapq.nsmallest(1, heap, key=_get_score)[0]

    backtrack = list()
    while head:
        backtrack.append(head.label)
        head = head.prev

    return list(reversed(backtrack[:-1]))
