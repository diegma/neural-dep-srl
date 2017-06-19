python3 -mnnet.run.srl.conll --data data/conll2009/CoNLL2009-ST-English-development.txt.jk > data/conll2009/dev_conll2009.json
python3 -mnnet.run.srl.conll --data data/conll2009/CoNLL2009-ST-evaluation-English.txt.jk  > data/conll2009/test_conll2009.json
python3 -mnnet.run.srl.conll --data data/conll2009/CoNLL2009-ST-English-train.txt.jk  > data/conll2009/train_conll2009.json
python3 -mnnet.run.srl.conll --data data/conll2009/CoNLL2009-ST-evaluation-English-ood.txt.jk > data/conll2009/ood_conll2009.json

python3 -mnnet.run.srl.training_sample --data data/conll2009/train_conll2009.json --frames data/nombank_descriptions-1.0+prop3.1.json > conll2009.train
python3 -mnnet.run.srl.training_sample --data data/conll2009/ood_conll2009.json --frames data/nombank_descriptions-1.0+prop3.1.json > conll2009.ood

python3 -mnnet.run.srl.training_sample --data data/conll2009/test_conll2009.json --frames data/nombank_descriptions-1.0+prop3.1.json > conll2009.test
python3 -mnnet.run.srl.training_sample --data data/conll2009/dev_conll2009.json --frames data/nombank_descriptions-1.0+prop3.1.json > conll2009.dev

cat conll2009.ood conll2009.test conll2009.train conll2009.dev conll2009.test > conll2009.combined

cut -f3 conll2009.combined | python3 -mnnet.ml.voc --tokenizer space >pos.voc.conll2009
cut -f6 conll2009.combined | python3 -mnnet.ml.voc --tokenizer space >frames.voc.conll2009
cut -f2 conll2009.train | python3 -mnnet.ml.voc --tokenizer space >words.voc_unk.conll2009
cut -f2 conll2009.train | python3 -mnnet.ml.voc -f --tokenizer space >freq.voc_unk.conll2009
cut -f10 conll2009.combined | python3 -mnnet.ml.voc --tokenizer space >labels.voc.conll2009
cut -f2 conll2009.combined | python3 -mnnet.ml.voc --tokenizer space >words.voc.conll2009

python3 -mnnet.run.srl.glove_select_srl words.voc.conll2009 data/sskip.100.vectors > word_embeddings_proper.sskip.conll2009.txt
cut -f1 word_embeddings_proper.sskip.conll2009.txt > p.words.voc_sskip.conll2009


