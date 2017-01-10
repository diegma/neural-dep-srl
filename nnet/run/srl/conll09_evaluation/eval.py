from subprocess import check_output
import os
import sys


def do_eval(subset_name, prediction, data_dir):

    def get_path(name):
        return os.path.join(data_dir, name)

    def get_proper_format(separate_predicate_list):
        with open('prediction_09.txt', 'w') as out:
            sent = ''
            sent_annotation = []
            for pred in separate_predicate_list:
                if sent == ' '.join([a[0] for a in pred]):
                    annotation = []
                    for t in range(len(pred)):
                        label = pred[t][1]
                        if label == 'O':
                            label = '_'
                        if t == pred[0][-2]:
                            annotation.append(('Y', label))
                        else:
                            annotation.append(('_', label))
                    sent_annotation[-1].append(
                        (annotation, pred[0][-2], pred[0][-1]))
                else:
                    if sent != '':
                        sent_annotation[-1] = sorted(sent_annotation[-1],
                                                     key=lambda x: int(x[1]))
                        for i, token in enumerate(sent_annotation[-1][0][0]):
                            single_token = []
                            for j in range(len(sent_annotation[-1])):
                                single_token.append(
                                    sent_annotation[-1][j][0][i][1])

                            out.write('\t'.join(single_token) + '\n')
                        out.write('\n')

                    annotation = []
                    sent = ' '.join([a[0] for a in pred])
                    for t in range(len(pred)):
                        label = pred[t][1]
                        if label == 'O':
                            label = '_'
                        if t == pred[0][-2]:
                            annotation.append(('Y', label))
                        else:
                            annotation.append(('_', label))
                    sent_annotation.append(
                        [(annotation, pred[0][-2], pred[0][-1])])

            sent_annotation[-1] = sorted(sent_annotation[-1],
                                         key=lambda x: int(x[1]))
            for i, token in enumerate(sent_annotation[-1][0][0]):
                single_token = []
                for j in range(len(sent_annotation[-1])):
                    single_token.append(sent_annotation[-1][j][0][i][1])

                out.write('\t'.join(single_token) + '\n')
            out.write('\n')
    get_proper_format(prediction)

    script = get_path('official_scripts/conll09/eval09.pl')
    gold_data = get_path('conll2009/eval/%s-set_for_eval_gold' % subset_name)

    eval_script_args = [script, '-g', gold_data, '-s' 'prediction_09.paste']

    try:
        DEVNULL = open(os.devnull, 'wb')

        past_script = ['paste', get_path(
            'conll2009/eval/%s-set_for_eval_ppred' % subset_name),
                       'prediction_09.txt']
        out_paste = check_output(past_script, stderr=DEVNULL)
        out_paste = out_paste.decode('utf-8')
        open('prediction_09.paste', 'w').write(out_paste)

        out = check_output(eval_script_args, stderr=DEVNULL)
        out = out.decode('utf-8')

        open('eval09.out', 'w').write(out)
        results = out.strip().split('\n')
        precision = results[7].split(' %')[0].split('= ')[1]
        recall = results[8].split(' %')[0].split('= ')[1]
        f1 = results[9].split()[2]
        print('--------------------------------------------------',
              file=sys.stderr)
        print('Official script results:', file=sys.stderr)
        print('Precision: ' + precision, 'recall: ' + recall, 'F1: ' + f1,
              file=sys.stderr)
        print('--------------------------------------------------',
              file=sys.stderr)
        return float(precision), float(recall), float(f1)
    except:
        print('--------------------------------------------------',
              file=sys.stderr)
        print('There has been some error with the official script',
              file=sys.stderr)
        print('Try next iteration :) ', file=sys.stderr)
        print('--------------------------------------------------',
              file=sys.stderr)
        return float(0.0), float(0.0), float(0.0)
