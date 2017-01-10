# An entry module                                                __INIT__.PY #

import sys
if sys.version_info[0] < 3:
    msg = "Sorry, 'ml' module is Python 3 only :("
    raise NotImplementedError(msg)

from nnet.ml.formats import create_tokenizer
from nnet.ml.formats import create_parser
