from collections import defaultdict
from nnet.util import *
from nnet.models.model import *
from copy import deepcopy


def train(model, train_set, optimizers, epochs, converter, tester, tickers,
          threshold, dbg_print_rate):
    batch_count, sample_count = 0, 0

    counter = 0
    best_model = deepcopy(model)

    log('running test with initial params...')
    tester.run(model)

    for e in range(epochs):
        tic = time.time()

        for batch in train_set.batches():
            record_ids, batch = zip(*batch)

            batch_count += 1
            sample_count += len(batch)

            model_input = converter(batch)

            bsize = len(record_ids)

            # print('loss', model.compute_loss(*model_input))
            if batch_count % dbg_print_rate == 0:
                log('[epoch %i, %i * %i] mean running loss = %f' %
                    (e, batch_count, len(batch),
                     np.mean(model.compute_loss(*model_input))))
                dash_line()

                best_loss_so_far = tester.best

                loss, errors, errors_w = tester.run(model)

                if loss < best_loss_so_far:
                    update_model(best_model, model)
                    counter = 0
                else:
                    counter += 1

                if counter == threshold and len(optimizers) > 1:
                    counter = 0
                    update_model(model, best_model)
                    optimizers = optimizers[1:]
                    log(colour(
                        "coudn't beat best model for %i times, "
                        "backing off to finer optimizer: %s" %
                        (threshold, optimizers[0][0]), 'yellow'))

            for ticker in tickers:
                ticker.tick(batch_count=batch_count,
                            sample_count=sample_count,
                            epoch=e)

            optimizers[0][1](*model_input)

        tac = time.time()

        passed = tac - tic
        log("epoch %i took %f min (~%f sec per sample)" % (
           e, passed / 60, passed / sample_count
        ))
