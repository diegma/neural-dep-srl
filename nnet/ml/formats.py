from functools import partial

def _ngrams(text, n):
    for i in range(len(text)):
        yield text[i:i + n]


def separator_tok(s, d):
    return s.split(d)


def char_tok(s):
    return s


def ngram_tok(s, order):
    ngrams = [s[i:i+order] for i in range(len(s))]
    ngrams = [['^'] * (order - 1) + item for item in ngrams]
    return ngrams


space_tok = partial(separator_tok, d=' ')
tab_tok = partial(separator_tok, d='\t')
hash_tok = partial(separator_tok, d='#')
bar_tok = partial(separator_tok, d='|')
csv_tok = partial(separator_tok, d=',')

bigram_tok = partial(ngram_tok, order=2)
trigram_tok = partial(ngram_tok, order=3)


def create_tokenizer(name):
    toks = {
        'delim': separator_tok,
        'char': char_tok,
        'ngram': ngram_tok,
        'bigram': bigram_tok,
        'space': space_tok,
        'tab': tab_tok,
        'hash': hash_tok,
        'bar': bar_tok,
        'csv': csv_tok
    }

    if name not in toks:
        raise NotImplementedError(name)

    return toks[name]


def classifier_parser(s):
    s = s.split('\t')
    if len(s) != 3 and len(s) != 4:
        raise RuntimeError("cannot parse '%s': %i fields instead of 3 or 4 "
                           % (s, len(s)))

    w = 1.0 if len(s) == 3 else float(s[3])
    
    res = s[0:2] + [int(s[2])] + [w]
    return res


def transducer_parser(s):
    s = s.split('\t')
    if len(s) != 2:
        raise RuntimeError(
            "cannot parse '%s': transduction data must have 2 fields") % s
    return s


def create_parser(name):
    parsers = {
        'classifier': classifier_parser,
        'transducer': transducer_parser
    }

    if name not in parsers:
        raise NotImplementedError(name)

    return parsers[name]

