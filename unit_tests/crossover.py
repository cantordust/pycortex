import cortex.utest as utest
import cortex.cortex as ctx
import cortex.random as Rand

import torch

def test_crossover():

    wheel = Rand.RouletteWheel()
    wheel.add('add_layer', 1)
    wheel.add('erase_layer', 1)
    wheel.add('add_node', 1)
    wheel.add('erase_node', 1)
    wheel.add('grow_kernel', 1)
    wheel.add('shrink_kernel', 1)

    nets = [ctx.Net() for _ in range(20)]

    for mut in range(10):
        for n in range(len(nets)):
            mutation = wheel.spin()

            net = nets[n]

            if mutation == 'add_layer':
                net.add_layer()

            elif mutation == 'add_node':
                net.add_nodes()

            elif mutation == 'grow_kernel':
                net.grow_kernel()

            if mutation == 'erase_layer':
                net.erase_layer()

            elif mutation == 'erase_node':
                net.erase_nodes()

            elif mutation == 'shrink_kernel':
                net.shrink_kernel()

            model = net.to('cpu')
            match = list(model(torch.randn(ctx.TrainBatchSize, *ctx.Net.Input.Shape)).size()) == [ctx.TrainBatchSize, net.layers[-1].get_output_nodes()]
            if not utest.pass_fail(match, "\tEvaluating the mutated network with random input..."):
                net.print()

    for p1 in range(len(nets)):
        for p2 in range(len(nets)):

            if p1 != p2:

                offspring = ctx.Net(_p1 = nets[p1], _p2 = nets[p2])

                model = offspring.to('cpu')
                match = list(model(torch.randn(ctx.TrainBatchSize, *ctx.Net.Input.Shape)).size()) == [ctx.TrainBatchSize, net.layers[-1].get_output_nodes()]
                if not utest.pass_fail(match, "\tEvaluating the offspring network with random input..."):
                    offspring.print()


if __name__ == '__main__':
    utest.run(test_crossover)