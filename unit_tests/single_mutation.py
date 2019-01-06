import cortex.utest as utest
import cortex.cortex as ctx

import torch

def test_single_mutation(_mut = 'add_layer'):

    net = ctx.Net()
    net.print('before_mutation.txt', True)
    if _mut == 'add_layer':
        success = net.add_layer()
    elif _mut == 'remove_layer':
        success = net.remove_layer()
    elif _mut == 'add_node':
        success = net.add_nodes()
    elif _mut == 'remove_node':
        success = net.remove_nodes()
    elif _mut == 'grow_kernel':
        success = net.grow_kernel()
    elif _mut == 'shrink_kernel':
        success = net.shrink_kernel()

    else:
        print("Invalid mutation type %r" % _mut)
        return

    assert(utest.pass_fail(success, "Mutating network..."))

    net.print('after_mutation.txt', True)

    model = net.to('cpu')

    tensor = torch.randn(ctx.TrainBatchSize, *ctx.Net.Input.Shape)
    print("Input size:", tensor.size())
    output = model(tensor)
    print(output)

if __name__ == '__main__':
    utest.run(test_single_mutation)
