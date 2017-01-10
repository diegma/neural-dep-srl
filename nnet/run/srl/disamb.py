from nnet.models.disamb import *
from nnet.run.runner import *
from nnet.ml.voc import *
from nnet.run.srl.run import *


class DisambErrorComputer():
    def __init__(self, converter, label_voc, word_voc, known_predicates):
        self.errors = 0
        self.converter = converter
        self.label_voc = label_voc
        self.word_voc = word_voc
        self.known_predicates = known_predicates
        self.log = open('tester.log', 'w')

    def apply_constraints(self, probs, true_label):
        sort = reversed(np.argsort(probs))
        for i in sort:
            label_w = self.label_voc.get_item(i)
            true_label_w = self.label_voc.get_item(true_label)
            
            if label_w.split('.')[0] == true_label_w.split('.')[0]:
                return i

    def compute(self, model, batch):
        errors, errors_w = 0, 0.0
        record_ids, batch = zip(*batch)
        model.test_mode_on()

        model_input = self.converter(batch)

        sent_batch, sent_mask, predicates, \
        predicate_marks_batch, predicate_pos, true_labels, pos_tags = model_input

        probs = model.predict(*model_input)

        for distribution, true_label, sent in zip(probs, true_labels, batch):
            best = self.apply_constraints(distribution, true_label)
            
            label_w = self.label_voc.get_item(best)
            pred_w = self.label_voc.get_item(true_label)

            if pred_w.split('.')[0] not in self.known_predicates.direct:
                # log(pred_w)
                label_w = pred_w.split('.')[0] + '.01'

            if label_w != pred_w:
                errors += 1
                errors_w += 1
            
            # print(label_w, pred_w, sent[1], ' '.join(sent[0]), file=self.log)

        loss = np.sum(model.compute_loss(*model_input))

        model.test_mode_off()
        self.errors += errors
        return loss, errors, errors_w

    def final(self):
        print('-----\n\n', file=self.log)
        errors = self.errors
        self.errors = 0
        return -errors


class DisambRunner(Runner):
    def __init__(self):
        super(DisambRunner, self).__init__()

        self.word_voc = create_voc('file', self.a.word_voc)
        self.label_voc = create_voc('file', self.a.label_voc)
        self.known_predicates = create_voc('file', self.a.known_predicates)
        self.pos_voc = create_voc('file', self.a.pos_voc)
        self.word_voc.add_unks()

    def add_special_args(self, parser):
        parser.add_argument("--word-voc", required=True)
        parser.add_argument("--label-voc", required=True)
        parser.add_argument("--pos-voc", required=True)
        parser.add_argument("--word-embeddings", required=True)
        parser.add_argument("--known-predicates", required=True)
        parser.add_argument("--hps", help="model hyperparams")

    def get_parser(self):
        def parse(record):
            dbg_header, sent, label, predicate_pos, pos_tags = record.strip().split('\t')
            sent = sent.split(' ')
            pos_tags = pos_tags.split(' ')
            predicate_pos = int(predicate_pos)
            predicate = sent[predicate_pos]

            predicate_marks = [1. if predicate_pos == i else 0. for i in range(len(sent))]

            return sent, predicate, predicate_marks, predicate_pos, label, pos_tags

        return parse

    def get_reader(self):
        return simple_reader

    def get_converter(self):
        def convert(batch):
            sents, predicates, predicate_marks, predicate_pos, labels, pos_tags = zip(*batch)
            sents = [self.word_voc.vocalize(sent) for sent in sents]
            pos_tags = [self.pos_voc.vocalize(pos_tag) for pos_tag in pos_tags]
            sent_batch, sent_mask = mask_batch(sents)
            pos_batch, pos_mask = mask_batch(pos_tags)

            predicates = self.word_voc.vocalize(predicates)
            predicate_marks_batch, _ = mask_batch(predicate_marks)
            predicate_marks_batch = np.array(predicate_marks_batch, dtype='float32')

            labels = self.label_voc.vocalize(labels)

            return sent_batch, sent_mask, predicates, predicate_marks_batch, predicate_pos, labels, pos_batch

        return convert

    def get_tester(self):
        converter = self.get_converter()
        computer = DisambErrorComputer(converter, self.label_voc, self.word_voc, self.known_predicates)
        corpus = Corpus(
            parser=self.get_parser(),
            batch_size=self.a.batch,
            path=self.a.test,
            reader=self.get_reader()
        )

        return ErrorRateTester(computer, self.a.out, corpus)

    def load_model(self):
        hps = eval(self.a.hps)

        hps['vword'] = self.word_voc.size()
        hps['vlabel'] = self.label_voc.size()
        hps['vpos'] = self.pos_voc.size()
        hps['word_embeddings'] = parse_word_embeddings(self.a.word_embeddings)

        return PredicateSenseDisamb(hps)


if __name__ == '__main__':
    DisambRunner().run()
