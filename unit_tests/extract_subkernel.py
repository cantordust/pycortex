import cortex.utest as utest
import cortex.cortex as ctx

def test_extract_subkernel(_layer_shape = [1,9,9],
                           _input_shape = [1, 28, 28],
                           _node_index = 0,
                           _patch_size = [3, 3]):

    layer = ctx.Layer(ctx.Layer.Def(_layer_shape), _input_shape)

    layer.print()

    sub = layer.extract_patch(_node_index, _patch_size)

    print(sub)

if __name__ == '__main__':
    utest.run(test_extract_subkernel)