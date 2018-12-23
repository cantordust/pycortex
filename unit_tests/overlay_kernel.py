import cortex.utest as utest
import cortex.cortex as ctx

def test_overlay_kernels():

    ctx.Net.Input.Shape = [1, 28, 28]
    ctx.Net.Init.Layers = [ctx.Layer.Def([5, 0, 0])]

    layer = ctx.Layer(ctx.Net.Init.Layers[0], ctx.Net.Input.Shape)
    print("================[ Initial layer ]================")
    layer.print()
    print("Weight tensor:\n", layer.weight)

    layer.overlay_kernels()
    print("================[ After overlaying kernels ]================")
    layer.print()
    print("Weight tensor:\n", layer.weight)

if __name__ == '__main__':
    utest.run(test_overlay_kernels)