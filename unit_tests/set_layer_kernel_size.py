import cortex.utest as utest
import cortex.cortex as ctx

def test_set_layer_kernel_size(_shape = [10, 0, 0]):

    print("Initial layer shape:", _shape)
    ctx.Net.Init.Layers = [ctx.Layer.Def(_shape)]

    net = ctx.Net()

    net.layers[0].print()

if __name__ == '__main__':
    utest.run(test_set_layer_kernel_size)
    utest.run(test_set_layer_kernel_size, [10, 3, 0])
    utest.run(test_set_layer_kernel_size, [10, 0, 3])
    utest.run(test_set_layer_kernel_size, [10, 3, 3])
