import cortex.random as Rand

class Ring(object):

    def __init__(self, _capacity):
        self.buffer = [None] * _capacity
        self.position = 0

    def push(self, _elem):
        self.buffer[self.position] = _elem
        self.position = (self.position + 1) % len(self.buffer)

    def sample(self, _batch_size, _weights):
        batch = []
        for n in range(_batch_size):
            batch.append(self.buffer[Rand.roulette(len(self.buffer), _weights)])

        return batch

    def __len__(self):
        return len(self.buffer)
