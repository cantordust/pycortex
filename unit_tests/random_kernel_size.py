import cortex.utest as utest
import cortex.random as Rand

import math

def test_random_kernel_size(_max = 28,
                            _draws = 1000):

    wheel = Rand.RouletteWheel()

    for i in range(1,_max):
       if i % 2 == 1:
            wheel.add(i, math.exp(-i))

    kernels = {}
    for draw in range(_draws):
        k = wheel.spin()

        if k not in kernels.keys():
            kernels[k] = 0
        kernels[k] += 1

    for key in sorted(kernels):
        print(key, ":", kernels[key])

if __name__ == '__main__':
    utest.run(test_random_kernel_size)