import cortex.utest as utest
import cortex.cortex as ctx

def test_overlay_kernels():

    ctx.cn.Net.Input.Shape = [1, 28, 28]
    ctx.cn.Net.Init.Layers = [ctx.cl.Layer.Def([5, 0, 0])]

    layer = ctx.cl.Layer(ctx.cn.Net.Init.Layers[0], ctx.cn.Net.Input.Shape)
    print("================[ Initial layer ]================")
    layer.print()
    print("Weight tensor:\n", layer.weight)

    layer.overlay_kernels()
    print("================[ After overlaying kernels ]================")
    layer.print()
    print("Weight tensor:\n", layer.weight)


    wheel = ctx.Rand.RouletteWheel()
    wheel.add(1, 1)
    wheel.add(-1, 1)

    print("================[ Resizing kernels ]================")

    attempts = 5
    successes = 0

    while successes < attempts:
        node = ctx.Rand.uint(0,len(layer.nodes))
        dim = ctx.Rand.uint(0,2)
        diff = wheel.spin()

        print("================[ Resizing kernel ]================")
        print('\n>>> Resizing dimension {} of kernel {} by {}'.format(dim, node, diff ))

        if (diff < 0 and
            list(layer.nodes[node].size())[dim + 1] < abs(diff) + 1):
            print('\tFailed: dimension too small')
            continue

        print('Before:\n{}'.format(layer.nodes[node]))
        layer.resize_kernel(node, {dim: diff})
        print('After:\n{}'.format(layer.nodes[node]))
        successes += 1

if __name__ == '__main__':
    utest.run(test_overlay_kernels)
