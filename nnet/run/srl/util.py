import sys

# Span = namedtuple('Span', ['name', 'begin', 'end'])


def get_spans(labels):
    spans = []
    span = []
    for i, label in enumerate(labels):
        if label.startswith('B-'):
            if span and span[0][0] != 'V':
                spans.append(span)
            span = [label[2:], i, i]
        elif label.startswith('I-'):
            if span:
                if label[2:] == span[0]:
                    span[2] = i
                else:
                    if span[0][0] != 'V':
                        spans.append(span)
                    span = [label[2:], i, i]
            else:
                span = [label[2:], i, i]
        else:
            if span and span[0][0] != 'V':
                spans.append(span)
            span = []
    if span:
        spans.append(span)
    return spans


def evaluate(predicted_labels, true_labels):
    def count_spans(spans):
        total = 0
        for span in spans:
            # if not span[0].startswith('C'):
            total += 1
        return total

    p_total, r_total, correct = 0., 0., 0.
    for predicted, true in zip(predicted_labels, true_labels):
        y_spans = get_spans(predicted)
        d_spans = get_spans(true)
        p_total += count_spans(y_spans)
        r_total += count_spans(d_spans)

        for y_span in y_spans:
            # if y_span[0].startswith('C'):
            #    continue
            if y_span in d_spans:
                correct += 1.

    if p_total > 0:
        p = correct / p_total
    else:
        p = 0.
    if r_total > 0:
        r = correct / r_total
    else:
        r = 0.
    if p + r > 0:
        f = (2 * p * r) / (p + r)
    else:
        f = 0.

    print('\tProps: %d\tP total: %f\tR total: %f\tCorrect: %f' % (
        len(predicted_labels), p_total, r_total, correct), file=sys.stderr)
    print('\tPrecision: %f\tRecall: %f\tF1: %f' % (p, r, f), file=sys.stderr)

    return f

def frame_data(data):
    import collections
    data = collections.OrderedDict(sorted(data.items()))
    for doc in data:
        for sentence in data[doc]:
            for field in data[doc][sentence]:
                if field.startswith('f_'):
                    for frame_instance in data[doc][sentence][field]:
                        yield doc, sentence, field[2:], frame_instance