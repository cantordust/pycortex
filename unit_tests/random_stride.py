#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cortex.utest as utest
import cortex.random as Rand
import cortex.functions as Func

def get_random_stride(_input_shape,
                      _stride = []): # Pre-determined stride. Dimensions with value 0 are populated with random values.

    wheel = Rand.RouletteWheel()

    print('Input shape: {}'.format(_input_shape))

    strides = []
    # Possible strides.
    for dim, radius in enumerate(_input_shape[1:]):

        if (len(_stride) > 0 and
            _stride[dim] > 0):
            stride = [_stride[dim]]

        else:
            if radius <= 1:
                stride = [1]

            else:
                stride = [s for s in range(1, radius // 2 + 1)]

        if len(strides) == 0:
            strides = [[s] for s in stride]

        else:
            new_strides = []

            for old_stride in strides:
                for new_stride in stride:
                    new_strides.append([*old_stride, new_stride])

            strides = new_strides

        print('Stride: {}'.format(stride))
        print('Strides: {}'.format(strides))

    for s in strides:
        wheel.add(s, Func.exp_prod(s))

#        for idx in range(len(wheel.elements)):
#            print(wheel.elements[idx], "\t", wheel.weights[Rand.WeightType.Raw][idx], "\t", wheel.weights[Rand.WeightType.Inverse][idx])

    return wheel.spin()

if __name__ == '__main__':

    input_shape = [3, 32, 32]

    print('Random stride: {}'.format(get_random_stride(input_shape)))
