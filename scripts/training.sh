THEANO_FLAGS='device=gpu0' python3 -mnnet.run.srl.run \
--train conll2009.train \
--test conll2009.dev \
--data_partition dev \
--batch 100 \
--freq-voc freq.voc_unk.conll2009 \
--word-voc words.voc_unk.conll2009 \
--p-word-voc p.words.voc_sskip.conll2009 \
--role-voc labels.voc.conll2009 \
--frame-voc frames.voc.conll2009 \
--pos-voc pos.voc.conll2009 \
--word-embeddings word_embeddings_proper.sskip.conll2009.txt \
--dbg-print-rate 200 \
--backoff-threshold 30 \
--rate 0.01 \
--optimizer adam \
--eval-dir ./data/ \
--soft-warnings \
--epochs 15 \
--out conll2009_rm0_pl_a.25_sskip_h512_d.0_l4 \
--hps "{'id': 1, 'sent_edim': 100, 'sent_hdim': 512, \
'frame_edim': 128, 'role_edim': 128, 'pos_edim': 16, 'rec_layers': 4, 'gc_layers': 0, \
'pos': True, 'rm':0, 'alpha': 0.25, \
'p_lemmas':True}"
