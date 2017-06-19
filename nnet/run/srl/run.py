from nnet.models.srl import *
from nnet.run.runner import *
from nnet.ml.voc import *
from nnet.run.srl.run import *
from nnet.run.srl.util import *
from nnet.run.srl.decoder import *
from functools import partial

from nnet.run.srl.read_dependency import get_adj
import nnet.run.srl.conll09_evaluation.eval


def make_local_voc(labels):
    return {i: label for i, label in enumerate(labels)}


def bio_reader(record):
    dbg_header, sent, pos_tags, dep_parsing, degree, frame, target, f_lemmas, f_targets, labels_voc, labels = record.split(
        '\t')
    labels_voc = labels_voc.split(' ')

    frame = [frame] * len(labels_voc)
    words = []
    for word in sent.split(' '):
        words.append(word)
        

    pos_tags = pos_tags.split(' ')
    labels = labels.split(' ')

    assert (len(words) == len(labels))

    local_voc = {v: k for k, v in make_local_voc(labels_voc).items()}
    labels = [local_voc[label] for label in labels]

    dep_parsing = dep_parsing.split()
    dep_parsing = [p.split('|') for p in dep_parsing]
    dep_parsing = [(p[0], int(p[1]), int(p[2])) for p in dep_parsing]

    f_lemmas = f_lemmas.split(' ')
    f_targets = f_targets.split(' ')
    return dbg_header, words, pos_tags, dep_parsing, np.int32(degree), frame, \
           np.int32(target), f_lemmas, np.int32(f_targets), labels_voc, labels


class BioSrlErrorComputer:
    def __init__(self, converter, do_labeling, data_partition,
                 eval_dir):
        self.converter = converter
        self.do_labeling = do_labeling
        self.data_partition = data_partition
        self.labels1 = []
        self.labels2 = []
        self.predictions = []
        self.eval_dir = eval_dir

        self.do_eval = nnet.run.srl.conll09_evaluation.eval.do_eval


    def pprint(self, record, voc, predictions, true_labels):
        n = 2

        sent = record[1]

        predictions = [p[:len(voc.items())] for p in predictions.tolist()]
        predictions = predictions[:len(sent)]

        constraints = []

        best_labeling = constrained_decoder(voc, predictions, 100, constraints)
        info = list()

        for word, prediction, true_label, best in zip(sent, predictions,
                                                      true_labels,
                                                      best_labeling):
            nbest = sorted(range(len(prediction)),
                           key=lambda x: -prediction[x])

            nbest = nbest[:n]
            probs = [prediction[l] for l in nbest]

            labels = [voc[label] for label in nbest if label in voc]
            labels = ' '.join(labels)

            info.append((word, labels, probs, voc[true_label], best))

        return info

    def compute(self, model, batch):
        errors, errors_w = 0, 0.0
        record_ids, batch = zip(*batch)

        model.test_mode_on()

        model_input = self.converter(batch)

        sent, \
        p_sent, \
        pos_tags,\
        sent_mask, \
        targets, \
        frames, \
        labels_voc, \
        labels_voc_mask, \
        freq, \
        region_mark, sent_pred_lemmas_idx, \
        adj_arcs_in, adj_arcs_out, adj_lab_in, adj_lab_out, \
        mask_in, mask_out, mask_loop, \
        true_labels \
            = model_input

        predictions = model.predict(*model_input[:-1])

        labels = np.argmax(predictions, axis=1)
        labels = np.reshape(labels, sent.shape)

        predictions = np.reshape(predictions,
                                 (sent.shape[0], sent.shape[1],
                                  labels_voc.shape[1]))

        for i, sent_labels in enumerate(labels):
            labels_voc = batch[i][-2]
            local_voc = make_local_voc(labels_voc)
            info = self.pprint(batch[i], local_voc, predictions[i],
                               true_labels[i])

            self.labels1.append([x[4] for x in info])
            self.labels2.append([x[3] for x in info])

            if self.do_labeling:
                sentence = []
                for word, label, probs, true, best in info:
                    single_pred = "%10s\t%10s\t%20s\t%40s\t%10s\t%10s\t%10s" % (
                        word, best, label, probs, true, batch[i][4],
                        batch[i][3][0])
                    sentence.append(
                        (word, best, label, probs, true, batch[i][4],
                         batch[i][3][0]))
                    print(single_pred)
                print('\n')
                self.predictions.append(sentence)
            else:
                sentence = []
                for word, label, probs, true, best in info:

                    sentence.append(
                        (word, best, label, probs, true, batch[i][4], batch[i][3][0]))
                self.predictions.append(sentence)

            for word, label, probs, true, best in info:
                if true != best:
                    errors += 1
                    errors_w += 1 / len(info)

        loss = np.sum(model.compute_loss(*model_input))

        model.test_mode_off()

        return errors, errors, errors_w

    def final(self):
        if self.do_eval:
            results = self.do_eval(self.data_partition, self.predictions,
                                   self.eval_dir)
        else:
            evaluate(self.labels1, self.labels2)
        self.predictions = []
        self.labels1, self.labels2 = [], []
        return results[2]


class SRLRunner(Runner):
    def __init__(self):
        super(SRLRunner, self).__init__()

        self.word_voc = create_voc('file', self.a.word_voc)
        self.word_voc.add_unks()
        self.freq_voc = frequency_voc(self.a.freq_voc)
        self.p_word_voc = create_voc('file', self.a.p_word_voc)
        self.p_word_voc.add_unks()
        self.role_voc = create_voc('file', self.a.role_voc)
        self.frame_voc = create_voc('file', self.a.frame_voc)
        self.pos_voc = create_voc('file', self.a.pos_voc)

    def add_special_args(self, parser):
        parser.add_argument(
            "--word-voc", required=True)
        parser.add_argument(
            "--p-word-voc", required=True)
        parser.add_argument(
            "--freq-voc", required=True)
        parser.add_argument(
            "--role-voc", required=True)
        parser.add_argument(
            "--frame-voc", required=True)
        parser.add_argument(
            "--pos-voc", required=True
        )
        parser.add_argument(
            "--word-embeddings", required=True
        )
        parser.add_argument(
            "--data_partition", required=True
        )
        parser.add_argument(
            "--hps", help="model hyperparams", required=False
        )
        parser.add_argument(
            "--eval-dir", help="path to dir with eval data and scripts",
            required=True
        )


    def get_parser(self):
        return partial(bio_reader)

    def get_reader(self):
        return simple_reader

    def get_converter(self):
        def bio_converter(batch):
            headers, sent_, pos_tags, dep_parsing, degree, frames, \
            targets, f_lemmas, f_targets, labels_voc, labels = list(
                zip(*batch))

            sent = [self.word_voc.vocalize(w) for w in sent_]
            p_sent = [self.p_word_voc.vocalize(w) for w in sent_]
            freq = [[self.freq_voc[self.word_voc.direct[i]] if
                     self.word_voc.direct[i] != '_UNK' else 0 for i in w] for
                    w
                    in sent]

            pos_tags = [self.pos_voc.vocalize(w) for w in pos_tags]
            frames = [self.frame_voc.vocalize(f) for f in frames]
            labels_voc = [self.role_voc.vocalize(r) for r in labels_voc]

            lemmas_idx = [self.frame_voc.vocalize(f) for f in f_lemmas]

            adj_arcs_in, adj_arcs_out, adj_lab_in, adj_lab_out, \
            mask_in, mask_out, mask_loop = get_adj(dep_parsing, degree)

            sent_batch, sent_mask = mask_batch(sent)
            p_sent_batch, _ = mask_batch(p_sent)
            freq_batch, _ = mask_batch(freq)
            freq_batch = freq_batch.astype(dtype='float32')

            pos_batch, _ = mask_batch(pos_tags)
            labels_voc_batch, labels_voc_mask = mask_batch(labels_voc)
            labels_batch, _ = mask_batch(labels)
            frames_batch, _ = mask_batch(frames)

            region_mark = np.zeros(sent_batch.shape, dtype='float32')
            hps = eval(self.a.hps)
            rm = hps['rm']
            if rm >= 0:
                for r, row in enumerate(region_mark):
                    for c, column in enumerate(row):
                        if targets[r] - rm <= c <= targets[r] + rm:
                            region_mark[r][c] = 1


            sent_pred_lemmas_idx = np.zeros(sent_batch.shape, dtype='int32')
            for r, row in enumerate(sent_pred_lemmas_idx):
                for c, column in enumerate(row):
                    for t, tar in enumerate(f_targets[r]):
                        if tar == c:
                            sent_pred_lemmas_idx[r][c] = lemmas_idx[r][t]

            sent_pred_lemmas_idx = np.array(sent_pred_lemmas_idx, dtype='int32')

            assert (sent_batch.shape == sent_mask.shape)
            assert (
                frames_batch.shape == labels_voc_batch.shape == labels_voc_mask.shape)
            assert (labels_batch.shape == sent_batch.shape)

            return sent_batch, p_sent_batch, pos_batch, sent_mask, targets, frames_batch, \
                   labels_voc_batch, \
                   labels_voc_mask, freq_batch, \
                   region_mark, \
                   sent_pred_lemmas_idx,\
                   adj_arcs_in, adj_arcs_out, adj_lab_in, adj_lab_out, \
                   mask_in, mask_out, mask_loop, \
                   labels_batch

        return bio_converter

    def get_tester(self):
        converter = self.get_converter()
        computer = BioSrlErrorComputer(
            converter, self.a.test_only, self.a.data_partition,
            self.a.eval_dir)
        corpus = Corpus(
            parser=partial(bio_reader),
            batch_size=self.a.batch,
            path=self.a.test,
            reader=self.get_reader()
        )

        return ErrorRateTester(computer, self.a.out, corpus)

    def load_model(self):
        hps = eval(self.a.hps)

        hps['vframe'] = self.frame_voc.size()
        hps['vword'] = self.word_voc.size()
        hps['vbio'] = self.role_voc.size()
        hps['vpos'] = self.pos_voc.size()
        hps['word_embeddings'] = parse_word_embeddings(self.a.word_embeddings)
        hps['in_arcs']=True
        hps['out_arcs']=True

        return BioSRL(hps)


if __name__ == '__main__':
    SRLRunner().run()
