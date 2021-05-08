class Ring(object):
    def __init__(self, _capacity):
        self.size = 0
        self.head = 0
        self.capacity = _capacity
        self.data = [None for _ in range(self.capacity)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.head + idx) % self.size]

    def append(self, _elem):
        if self.size < self.capacity:
            # Increment the size
            self.size += 1

        self.head = (self.head + 1) % self.size
        self.data[self.head] = _elem

    def dump(self):
        data = []
        head = self.head
        for _ in range(self.size):
            head = (head + 1) % self.size
            data.append(self.data[head])

        return data

    def zp_dump(self):
        data = [0.0] * self.capacity
        head = self.head
        for pos in range(self.size):
            head = (head + 1) % self.size
            data[self.capacity - self.size + pos] = self.data[head]

        return data
