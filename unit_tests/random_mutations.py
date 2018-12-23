import cortex.utest as utest
import cortex.cortex as ctx
import cortex.random as Rand

import torch

def test_random_mutations():

    for i in range(10):

        wheel = Rand.RouletteWheel()
        wheel.add('add_layer', 1)
        wheel.add('add_node', 1)
        wheel.add('grow_kernel', 1)

        original_net = ctx.Net(_isolated = True)

        # Clone the network
        net = ctx.Net(_p1 = original_net, _isolated = True)

        assert (utest.pass_fail(net.matches(original_net), "Comparing the original network with the cloned one..."))

        mutations = []

        for mut in range(100):
            mutation = wheel.spin()
            success = False
            if mutation == 'add_layer':
                success, layer = net.add_layer(_test = True)
                if success:
                    mutations.append((mutation, layer))
            elif mutation == 'add_node':
                success, layer, nodes = net.add_nodes()
                if success:
                    mutations.append((mutation, layer, nodes))
            elif mutation == 'grow_kernel':
                success, layer, kernel, delta = net.grow_kernel()
                if success:
                    mutations.append((mutation, layer, kernel, delta))

            if success:
                print("(", mut + 1, ") Mutation:", *mutations[-1])
            model = net.to('cpu')
            assert(model(torch.randn(ctx.TrainBatchSize, *ctx.Net.Input.Shape)).size())

        print("\n==============[ Reversing mutations ]==============\n")

        for mut in range(len(mutations)):
            mutation = mutations[len(mutations) - mut - 1]
            success = False
            if mutation[0] == 'add_layer':
                success, layer = net.erase_layer(mutation[1])
                #if success:
                    #print("Layer", layer, "erased")
            elif mutation[0] == 'add_node':
                success, layer, node = net.erase_nodes(mutation[1], _node_indices = mutation[2])
                #if success:
                    #print("Node", *nodes, "erased from layer", layer)
            elif mutation[0] == 'grow_kernel':
                success, layer, kernel, delta = net.shrink_kernel(mutation[1], mutation[2], mutation[3])
                #if success:
                    #print("Dimension", *delta.keys(), "of kernel", kernel, "in layer", layer, "decreased by", abs(*delta.values()))

            assert (utest.pass_fail(success, "Reversing mutation", len(mutations) - mut, "(", mutation, ")..."))
            model = net.to('cpu')
            output = model(torch.randn(ctx.TrainBatchSize, *ctx.Net.Input.Shape))

        assert (utest.pass_fail(net.matches(original_net), "Comparing the original network with the one with reversed mutations..."))

        print("======================[ Original network ]======================")
        original_net.print()

        print("======================[ Mutated network ]=======================")
        net.print()

        model1 = net.to('cpu')
        model2 = original_net.to('cpu')

        input1 = torch.randn(ctx.TrainBatchSize, *ctx.Net.Input.Shape)
        input2 = torch.zeros(input1.size())
        input2.data = input1.data

        print("Input1:", input1, ", size:", input1.size())
        print("Input2:", input2, ", size:", input2.size())

        output1 = model1(input1)
        output2 = model2(input2)

        assert(utest.pass_fail(torch.allclose(output1, output2), "Comparing the two outputs..."))

        print(output1)
        print(output2)

if __name__ == '__main__':
    utest.run(test_random_mutations)