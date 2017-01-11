# neural-dep-srl
This is the code for reproducing the experiments in the paper [A Simple and Accurate Syntax-Agnostic Neural Model for Dependency-based Semantic Role Labeling](https://arxiv.org/abs/1701.02593).

Dependencies
-----------
- [theano](http://deeplearning.net/software/theano/)
- [lasagne](http://lasagne.readthedocs.io/)

Semantic role labeling data processing
--------------
To run the model the first thing to do is create a dataset and all the files needed for the evaluation.

1) Place the CoNLL-2009 dataset files with the same format as in [here](https://ufal.mff.cuni.cz/conll2009-st/task-description.html) in data/conll2009/

2) Place the embedding file [sskip.100.vectors](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing) in data/

3) Run scripts/srl_preproc.sh in order to obtain the preprocessed data you need for training and testing the model.

4) Place the development test and ood files in /data/conll/eval/ and rename them respectively dev-set_for_eval_gold, test-set_for_eval_gold, ood-set_for_eval_gold.

5) Place the files dev-set_for_eval, test-set_for_eval, ood-set_for_eval in /data/conll/eval/ with only the first 12 columns and as 13th column put your predicted predicate sense, and rename the files respectively dev-set_for_eval_ppred, test-set_for_eval_ppred, ood-set_for_eval_ppred

Semantic role labeling training and testing
--------------
6) To train your model run scripts/train.sh

7) To test the trained model run scripts/train.sh

The hyper-parameters on the scripts are the ones with which we obtained the best results.

For any question, send us a mail at marcheggiani [at] uva [dot] nl or anton-fr [at] yandex-team [dot] ru .