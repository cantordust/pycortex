import cortex.utest as utest
import cortex.cortex as ctx
from copy import deepcopy as dcp

def test_overlay_kernels():

    ctx.cn.Net.Input.Shape = [1, 28, 28]
    ctx.cn.Net.Init.Layers = [ctx.cl.Layer.Def([5, 0, 0])]

    layer = ctx.cl.Layer(ctx.cn.Net.Init.Layers[0], ctx.cn.Net.Input.Shape)
    print('================[ Initial layer ]================')
    print(layer.as_str())
    print(f'Weight tensor: {layer.weight}')

    layer.overlay_kernels()
    print('================[ After overlaying kernels ]================')
    print(layer.as_str())
    print(f'Weight tensor: {layer.weight}')


    wheel = ctx.Rand.RouletteWheel()
    wheel.add(1, 1)
    wheel.add(-1, 1)

    print('================[ Resizing kernels ]================')

    attempts = 5
    successes = 0

    while successes < attempts:
        node = ctx.Rand.uint(0,len(layer.nodes))
        dim = ctx.Rand.uint(0,2)
        diff = wheel.spin()

        print('================[ Resizing kernel ]================')
        print(f'\n>>> Resizing dimension {dim} of kernel {node} by {diff}')

        node = dcp(layer.nodes[node])

        mut = layer.resize_kernel(node, dim, diff)

        if mut.success:
            print(f'''
                    Before:
                    {node}
                    After:
                    {layer.nodes[node]}
                    ''')

            successes += 1

if __name__ == '__main__':
    utest.run(test_overlay_kernels)
