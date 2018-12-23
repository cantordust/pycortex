import cortex.utest as utest
import torch.nn as tn

def test_layer_output_shape(_input_shape = [3, 32, 32],
                           _output_nodes = 10,
                           _kernel_size = (3, 3),
                           _padding = (),
                           _stride = (),
                           _dilation = ()):

    layer_types = {
        1: tn.Conv1d,
        2: tn.Conv2d,
        3: tn.Conv3d
        }

    if len(_kernel_size) == 0:
        layer = tn.Linear(_input_shape[0], _output_nodes)
    else:

        if len(_padding) == 0:
            _padding = tuple([dim // 2 for dim in _kernel_size])

        if len(_stride) == 0:
            _stride = tuple([1] * len(_kernel_size))

        if len(_dilation) == 0:
            _dilation = tuple([1] * len(_kernel_size))

        layer = layer_types[len(_kernel_size)](_input_shape[0], _output_nodes, _kernel_size, _stride, _padding, _dilation)

    layer_shape = list(layer.weight.size())

    del layer_shape[1] # Input channels

    print("Output nodes:", _output_nodes)
    print("Input shape:", _input_shape)
    print("Kernel size:", _kernel_size)
    print("Padding:", _padding)
    print("Striide:", _stride)
    print("Dilation:", _dilation)

    if len(_kernel_size) > 0:
        for dim in range(len(_kernel_size)):
            layer_shape[dim + 1] = (_input_shape[dim + 1] + 2 * _padding[dim] - _dilation[dim] * (layer_shape[dim + 1] - 1) - 1) // _stride[dim] + 1

    print("Output shape:", layer_shape)

if __name__ == '__main__':
    utest.run(test_layer_output_shape)