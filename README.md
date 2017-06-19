# neural-dep-srl
This is the code for used in the papers [A Simple and Accurate Syntax-Agnostic Neural Model for Dependency-based Semantic Role Labeling](https://arxiv.org/abs/1701.02593) and [Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling](https://arxiv.org/abs/1703.04826).

Dependencies
-----------
- python 3
- [theano 0.8.2](http://deeplearning.net/software/theano/)
- [lasagne 0.1](http://lasagne.readthedocs.io/)

Semantic role labeling data processing
--------------
To run the model the first thing to do is create a dataset and all the files needed for the evaluation.

1) Place the CoNLL-2009 dataset files with the same format as in [here](https://ufal.mff.cuni.cz/conll2009-st/task-description.html) in data/conll2009/

2) Place the embedding file [sskip.100.vectors](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing) in data/

3) Run scripts/srl_preproc.sh in order to obtain the preprocessed data you need for training and testing the model.

4) Place the development, test, and ood files in /data/conll/eval/ and rename them respectively dev-set_for_eval_gold, test-set_for_eval_gold, ood-set_for_eval_gold.

5) Place the dev, test, and ood files in /data/conll/eval/ with only the first 12 columns and as 13th column put your predicted predicate sense, and rename the files respectively dev-set_for_eval_ppred, test-set_for_eval_ppred, ood-set_for_eval_ppred

Semantic role labeling training and testing
--------------
6a. To train the sintax agnostic model run scripts/train.sh

6b. To train the model with the graph convolutional network over syntax run scripts/train_gcn.sh

7) To test the trained model run scripts/test.sh

The hyper-parameters on the scripts are the ones with which we obtained the best results.

For any question, send us a mail at marcheggiani [at] uva [dot] nl or anton-fr [at] yandex-team [dot] ru .