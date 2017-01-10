import itertools
import functools
import random


def simple_reader(data, batch_size):
    args = [iter(enumerate(data))] * batch_size
    for batch in itertools.zip_longest(*args, fillvalue=None):
        yield [s for s in batch if s is not None]


def homogeneous_reader_1(data, batch_size):
    buckets = sorted(enumerate(data), key=len)
    buckets = [list(bucket[1]) for bucket in
               itertools.groupby(buckets, key=len)]

    record_cnt = functools.reduce(lambda a, x: a + len(x), buckets, 0)

    for i in range(int(record_cnt / batch_size) + 1):
        pivot = idx = random.randint(0, len(buckets) - 1)
        way = 1 if random.randint(0, 1) == 1 else -1

        batch = list()

        size = batch_size
        while size:
            if idx < 0 or idx > len(buckets) - 1:
                way *= -1
                idx = pivot

            bucket = buckets[idx]
            cnt = min(size, len(bucket))
            batch += random.sample(bucket, cnt)
            size -= cnt

            idx += way

        yield batch


def homogeneous_reader_2(data, batch_size):
    data = sorted(enumerate(data), key=len)
    args = [iter(data)] * batch_size
    batches = list(itertools.zip_longest(*args, fillvalue=None))
    random.shuffle(batches)
    for batch in batches:
        batch = list(batch)
        random.shuffle(batch)
        yield [s for s in batch if s is not None]


class Corpus(object):
    def __init__(self, parser, batch_size, path, reader=simple_reader):
        self.parser = parser
        self.path = path
        self.batch_size = batch_size
        self._size = None
        self.reader = reader

    def batches(self):
        with open(self.path, 'r') as data:
            for batch in self.reader(data, self.batch_size):
                batch = [(id, self.parser(record[:-1])) for (id, record) in
                         batch]
                yield batch

    def get_batch_size(self):
        return self.batch_size

    def size(self):
        if self._size is None:
            for i, _ in enumerate(open(self.path, 'r')):
                pass
            self._size = i + 1
        return self._size
