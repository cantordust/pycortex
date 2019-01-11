import sys
import math
from copy import deepcopy as dcp

import torch
import torch.nn as tn
import torch.nn.functional as tnf

import cortex.functions as Func
import cortex.statistics as Stat
import cortex.random as Rand

import cortex.species as cs
import cortex.layer as cl

class Mutation:

    def __init__(self):
        self.success = False
        self.msg = ''

class Net(tn.Module):

    # Configuration
    class Input:
        Shape = []

    class Output:
        Shape = []

    class Init:
        Count = 8
        Layers = []

    class Max:
        Count = 16
        Age = 50

    # Static members
    ID = 0
    Champion = None
    Ecosystem = {}

    @staticmethod
    def reset():
        Net.ID = 0
        Net.Champion = None
        Net.Ecosystem = {}

    def __init__(self,
                 _empty = False,
                 _p1 = None,
                 _p2 = None,
                 _species = None,
                 _isolated = False):

        super(Net, self).__init__()

        from cortex.fitness import Fitness

        # Assign a network ID
        self.ID = 0

        # Species ID
        self.species_id = 0

        if not _isolated:
            # Increment the ID counter
            Net.ID += 1
            self.ID = Net.ID

            # Add this network to the ecosystem
            Net.Ecosystem[self.ID] = self

            if isinstance(_species, cs.Species):
                self.species_id = _species.ID
                cs.Species.Populations[self.species_id].nets.add(self.ID)

        # Initialise the age
        self.age = 0

        # Initialise the fitness
        self.fitness = Fitness()

        # Generate a module list
        self.layers = tn.ModuleList()

        if not _empty:

            if (isinstance(_p1, Net)):
                if (isinstance(_p2, Net)):
                    print('[Net {}] >>> Performing crossover with network {}...'.format(_p1.ID, _p2.ID))
                    self.crossover(_p1, _p2)
                else:
                    print('[Net {}] >>> Cloning...'.format(_p1.ID))
                    self.clone(_p1)

            else:

                layer_defs = _species.genome if isinstance(_species, cs.Species) else Net.Init.Layers

                if len(layer_defs) > 0:
                    for layer_index, layer_def in enumerate(layer_defs):
                        self.add_layer(_shape = layer_def.shape,
                                       _stride = layer_def.stride,
                                       _bias = layer_def.bias,
                                       _activation = layer_def.activation,
                                       _layer = layer_index)

                # Output layer
                self.add_layer(_shape = Net.Output.Shape,
                               _bias = cl.Layer.Bias,
                               _layer = len(self.layers))

        print(">>> Network", self.ID, "created")

    def matches(self,
                _other):

        if len(self.layers) != len(_other.layers):
#            print("\t>>> Different number of layers")
#            print("\t>>> Network 1:\n")
#            self.print()
#            print("\t>>> Network 2:\n")
#            _other.print()
            return False

        for layer_index in range(len(self.layers)):
            if not self.layers[layer_index].matches(_other.layers[layer_index]):
                return False

        return True

    def as_str(self,
               _layers = True):

        str = f'''
###################[ Network {self.ID} ]###################
>>> Fitness:
    Absolute: {self.fitness.absolute}
    Relative: {self.fitness.relative}
>>> Age: {self.age}
>>> Species: {self.species_id}
>>> Total parameters: {self.get_parameter_count()}
'''

        if _layers:
            for layer in self.layers:
                str += layer.as_str()
        return str

    def is_conv(self):

        # To determine whether this network is convolutional,
        # it is enough to check if the first layer is convolutional.
        if len(self.lyaers) > 0:
            return self.layers[0].is_conv

        return False

    def get_genome(self):

        genome = []
        for layer_index in range(len(self.layers) - 1):
            layer = self.layers[layer_index]
            shape = [0] * len(layer.get_output_shape())
            shape[0] = len(layer.nodes)

            genome.append(cl.Layer.Def(_shape = shape,
                                       _stride = layer.stride,
                                       _bias = layer.bias is not None,
                                       _activation = layer.activation,
                                       _role = layer.role))

        return genome

    def reindex(self):
        for index, layer in enumerate(self.layers):
            layer.index = index

            if index == len(self.layers) - 1:
                layer.role = 'output'
            else:
                layer.role = layer.op.__name__

    def get_input_shape(self,
                        _layer):
        """
        Compute the shape of this layer's input.
        """

        if _layer <= 0:
            return Net.Input.Shape

        assert _layer <= len(self.layers), "(get_input_shape) Invalid layer index %r (network %r contains %r layer(s))" % (_layer, self.ID, len(self.layers))

        return self.layers[_layer - 1].get_output_shape()

    def get_output_shape(self,
                         _layer):
        """
        Compute this layer's output shape based on the shape of the preceding layer
        (or the input shape if it is the first layer) and the layer's kernels.
        """

        assert _layer < len(self.layers), "(get_output_shape) Invalid layer index %r (network %r contains %r layers)" % (_layer, self.ID, len(self.layers))

        return self.layers[_layer].get_output_shape()

    def get_allowed_layer_shapes(self,
                                 _layer):

        """
        Check what layer shapes are allowed at this index
        """
        # The output layer has a predetermined shape
        if _layer >= len(self.layers):
            return [Net.Output.Shape]

        input_shape = [0] * len(self.get_input_shape(_layer))
        output_shape = [0] * len(self.get_output_shape(_layer))

        return [input_shape] if len(input_shape) == len(output_shape) else [input_shape, output_shape]

    def get_parameter_count(self):

        parameters = sum([param.numel() for param in self.parameters() if param.requires_grad])
#        print("Net", self.ID, "parameter count:", parameters)

        return parameters

    def get_relative_complexity(self):

        complexity_stat = Stat.SMAStat()
        for net in Net.Ecosystem.values():
            complexity_stat.update(net.get_parameter_count())

        return complexity_stat.get_offset(net.get_parameter_count())

    def add_layer(self,
                  _shape = [],
                  _stride = [],
                  _bias = None,
                  _activation = None,
                  _layer = None,
                  _test = False):

        mut =  Mutation()

        mut.layer = _layer
        mut.shape = _shape
        mut.stride = _stride
        mut.bias = _bias
        mut.activation = _activation

        if (mut.layer is None or
            len(mut.shape) == 0):

            wheel = Rand.RouletteWheel()

            for layer_index in range(len(self.layers)):

                # Check how many links we have to add and / or remove
                # to insert a layer of each allowed shape

                node_stat = Stat.SMAStat()
                for layer in self.layers:
                    node_stat.update(len(layer.nodes))

#                new_nodes = math.floor(_stats['nodes'].mean)
                for allowed_shape in self.get_allowed_layer_shapes(layer_index):

                    if (len(mut.shape) > 0 and
                        mut.shape[0] > 0):
                        new_nodes = mut.shape[0]
                    else:
                        new_nodes = Rand.uint(1, math.floor(node_stat.mean + 1))

                    input_shape = self.get_input_shape(layer_index)
                    new_layer_shape = dcp(allowed_shape)

                    # Set the number of output nodes
                    if _test:
                        # This is a special case used for unit tests.
                        # It is necessary in order to ensure that the mutation is reversible.
                        # TODO: Better unit test that doesn't require this special block.
                        new_layer_shape = [0] * len(self.get_input_shape(layer_index))
                        new_layer_shape[0] = self.layers[layer_index].get_input_nodes() + 1

                    elif new_layer_shape[0] == 0:
                        new_layer_shape[0] = new_nodes
#                        new_layer_shape[0] = 1

                    wheel.add((layer_index, new_layer_shape), 1)

            if wheel.is_empty():
                mut.msg = 'Empty roulette wheel'
                return mut

            random_layer, random_shape = wheel.spin()

            if mut.layer is None:
                mut.layer = random_layer

            if len(mut.shape) == 0:
                mut.shape = random_shape

        # Sanity check for the layer index
        if mut.layer > len(self.layers):
            #print("Invalid layer index", _layer)
            mut.msg = f'Invalid layer index ({mut.layer}): must be <= {len(self.layers)}'
            return mut

        # Create a layer definition if not provided
        layer_def = cl.Layer.Def(_shape = mut.shape,
                                 _stride = mut.stride,
                                 _bias = mut.bias,
                                 _activation = mut.activation)

        input_shape = self.get_input_shape(mut.layer)

        # Ensure that the layer roles are contiguous.
        # This can be done with a simple comparison of the input and output shapes.
        if (len(input_shape) < len(layer_def.shape) or                              # Attempting to add a conv layer above an FC one
            (mut.layer < len(self.layers) and
             len(layer_def.shape) < len(self.get_output_shape(mut.layer)))):     # Attempting to add an FC layer below a conv one

            mut.msg = f'Invalid layer size: Cannot add layer of shape {layer_def.shape} at position {mut.layer}'
            return mut

        # Create the new layer
        new_layer = cl.Layer(_layer_def = layer_def,
                             _input_shape = input_shape,
                             _index = mut.layer)

        # Rearrange the module stack if necessary
        if mut.layer == len(self.layers):
            self.layers.append(new_layer)

        else:
            new_ml = tn.ModuleList()

            for index in range(len(self.layers)):
                if index == mut.layer:
                    new_ml.append(new_layer)

                new_ml.append(self.layers[index])

            self.layers = new_ml

        # Adjust the input size of the following layers
        for layer_index in range(mut.layer + 1, len(self.layers)):
            self.layers[layer_index].adjust_input_size(_input_shape = self.get_input_shape(layer_index))

        self.reindex()

        mut.success = True
        mut.msg = f'Added layer {new_layer.as_str()}'

        return mut

    def remove_layer(self,
                    _layer = None):

        mut =  Mutation()

        mut.layer = _layer

        if len(self.layers) < 2:
            mut.msg = 'Not enough layers'
            return mut

        # Update the input size of the next layer (if there is one)
        if mut.layer is None:

            wheel = Rand.RouletteWheel()
            for layer_index in range(len(self.layers) - 1):
                wheel.add((layer_index,), 1)

            if wheel.is_empty():
                mut.msg = 'Empty roulette wheel'
                return mut

            mut.layer = wheel.spin()[0]

        # We cannot remove the output layer
        if mut.layer == len(self.layers) - 1:
            mut.msg = 'Attempted to remove the output layer'
            return mut

        mut.msg = f'Removed layer:\n{self.layers[mut.layer].as_str()}'

        # Remove the layer
        del self.layers[mut.layer]

        # Adjust the input size of the following layers
        for layer_index in range(mut.layer, len(self.layers)):
            self.layers[layer_index].adjust_input_size(_input_shape = self.get_input_shape(layer_index))

        self.reindex()

        mut.success = True

        return mut

    def add_nodes(self,
                  _layer = None,
                  _count = 1,
                  _max_radius = []):

        mut =  Mutation()

        mut.layer = _layer
        mut.count = _count
        mut.nodes = set()

        if mut.count <= 0:
            mut.msg = f'Invalid count {mut.count}'
            return mut

        if mut.layer is None:

            wheel = Rand.RouletteWheel()
            for layer_index in range(len(self.layers) - 1):
                wheel.add((layer_index,), 1)

            if wheel.is_empty():
                mut.msg = 'Empty roulette wheel'
                return mut

            mut.layer = wheel.spin()[0]

        for n in range(len(self.layers[mut.layer].nodes), len(self.layers[mut.layer].nodes) + mut.count):
            mut.nodes.add(n)

        mut.count = len(mut.nodes)

        # Sanity checks
        if len(self.layers) == 0:
            mut.msg = 'No layers found'
            return mut

        if mut.layer >= len(self.layers) - 1:
            mut.msg = 'Attempted to add nodes to the output layer'
            return mut

        # Add the nodes
        mut.success = self.layers[mut.layer].add_nodes(mut.count, _max_radius)

        # On success, adjust the input size of the next layer if there is one
        if mut.success:
            self.layers[mut.layer + 1].adjust_input_size(_input_shape = self.get_output_shape(mut.layer))
            mut.msg = f'Added node{"s" if mut.count > 1 else ""} {mut.nodes} to layer {mut.layer}'

        return mut

    def remove_nodes(self,
                    _layer = None,
                    _count = 1,
                    _nodes = set()):

        mut =  Mutation()

        mut.layer = _layer
        mut.count = _count
        mut.nodes = set()

        # Sanity checks
        if isinstance(_nodes, int):
            mut.nodes.add(_nodes)

        elif isinstance(_nodes, list):
            for node_index in _nodes:
                mut.nodes.add(node_index)

        if (mut.layer is None or
            len(mut.nodes) == 0):

            layer_wheel = Rand.RouletteWheel()

            for layer_index in range(len(self.layers) - 1):

                # Check if we can remove a node at all
                if len(self.layers[layer_index].nodes) > 1:
                    layer_wheel.add(layer_index, 1)

            if layer_wheel.is_empty():
                mut.msg = 'Empty layer roulette wheel'
                return mut

            layer_index = layer_wheel.spin()

            # Create a new wheel only for the nodes in the selected layer
            node_wheel = Rand.RouletteWheel()
            for node_index in range(len(self.layers[layer_index].nodes)):
                node_wheel.add(node_index, 1)

            if node_wheel.is_empty():
                mut.msg = 'Empty node roulette wheel'
                return mut

            # Store the layer index
            if mut.layer is None:
                mut.layer = layer_index

            # Populate the node indices
            if len(mut.nodes) == 0:
                # Check if we have enough nodes to work with
                if mut.count >= len(self.layers[mut.layer].nodes):
                    mut.msg = f'Attempted to remove {mut.count} nodes from a layer with {len(self.layers[mut.layer].nodes)} nodes'
                    return mut

                # Draw the necessary number of node indices
                while len(mut.nodes) < mut.count:
                    mut.nodes.add(node_wheel.pop())

        # Sanity check for the layer index
        if len(self.layers) == 0:
            mut.msg = 'No layers found'
            return mut

        if mut.layer >= len(self.layers) - 1:
            mut.msg = 'Attempted to remove nodes from the output layer'
            return mut

        # Sanity check for node indices.
        if len(mut.nodes) > 0:
            for node_index in mut.nodes:
                if (node_index < 0 or
                    node_index >= len(self.layers[mut.layer].nodes)):
                    mut.msg = f'Invalid node index {node_index}'
                    return mut

        mut.count = len(mut.nodes)

        if mut.count == 0:
            mut.msg = f'Invalid number of nodes to remove ({mut.count})'
            return mut

        # Remove the nodes
        mut.success = self.layers[mut.layer].remove_nodes(sorted(list(mut.nodes)))

        # On success, adjust the input size of the next layer if there is one
        if (mut.success and
            mut.layer < len(self.layers) - 1):
            self.layers[mut.layer + 1].adjust_input_size(_input_shape = self.get_output_shape(mut.layer),
                                                         _nodes = sorted(list(mut.nodes)))

            mut.msg = f'Removing node({"s" if mut.count > 1 else ""}) {mut.nodes} from layer {mut.layer}'

        return mut

    def resize_kernel(self,
                      _layer = None,
                      _node = None,
                      _dim = None,
                      _diff = None):

        mut =  Mutation()

        mut.layer = _layer
        mut.node = _node
        mut.dim = _dim
        mut.diff = _diff

        if (mut.layer is None or
            mut.node is None or
            mut.dim is None or
            mut.diff is None):

            wheel = Rand.RouletteWheel()

            if mut.layer is None:
                layers = [l for l in range(len(self.layers)) if self.layers[l].is_conv]
            elif (mut.layer < len(self.layers) and
                  self.layers[mut.layer].is_conv):
                layers = [mut.layer]
            else:
                layers = []

            if len(layers) == 0:
                mut.msg = 'Failed to find evolvable layers'
                return mut

            for layer in layers:

                if mut.node is None:
                    nodes = [n for n in range(len(self.layers[layer].nodes))]
                elif mut.node < len(self.layers[layer].nodes):
                    nodes = [mut.node]
                else:
                    nodes = []

                if len(layers) == 0:
                    mut.msg = 'Failed to find evolvable kernels'
                    return mut

                for node in nodes:

                    if mut.dim is None:
                        dims = [d for d in range(len(self.layers[layer].kernel_size))]
                    elif mut.dim < len(list(self.layers[layer].nodes.size()) - 1):
                        dims = [mut.dim]
                    else:
                        dims = []

                    if len(dims) == 0:
                        mut.msg = 'Failed to find evolvable kernel dimensions'
                        return mut

                    for dim in dims:
                        wheel.add((layer, node, dim), 1)

            while True:

                if wheel.is_empty():
                    mut.msg = 'Empty roulette wheel'
                    return mut

                mut.layer, mut.node, mut.dim = wheel.pop()

                grow_allowed = list(self.layers[mut.layer].nodes[mut.node].size())[mut.dim + 1] < self.get_input_shape(mut.layer)[mut.dim + 1] // 2
                shrink_allowed = list(self.layers[mut.layer].nodes[mut.node].size())[mut.dim + 1] > 1

                if (mut.diff < 0 and not shrink_allowed or
                    mut.diff > 0 and not grow_allowed):
                    continue

                if mut.diff is None:
                    if shrink_allowed:
                        if grow_allowed:
                            mut.diff = -1 if Rand.chance(0.5) else 1
                        else:
                            mut.diff = -1
                    elif grow_allowed:
                        mut.diff = 1

                if mut.diff is not None:
                    break

        # Resize the kernel
        mut.success = self.layers[mut.layer].resize_kernel(mut.node, mut.dim, mut.diff)

        if mut.success:
            mut.msg = f'Resized dimension {mut.dim} of kernel {mut.node} in layer {mut.layer} by {mut.diff}'

        return mut

    def resize_stride(self,
                      _layer = None,
                      _dim = None,
                      _diff = None):

        mut =  Mutation()

        mut.layer = _layer
        mut.dim = _dim
        mut.diff = _diff

        if (mut.layer is None or
            mut.dim is None or
            mut.diff is None):

            wheel = Rand.RouletteWheel()

            if mut.layer is None:
                layers = [l for l in range(len(self.layers)) if self.layers[l].is_conv]
            elif (mut.layer < len(self.layers) and
                  self.layers[mut.layer].is_conv):
                layers = [mut.layer]
            else:
                layers = []

            if len(layers) == 0:
                mut.msg = 'Failed to find evolvable layers'
                return mut

            for layer in layers:

                if mut.dim is None:
                    dims = [d for d in range(len(self.layers[layer].stride))]
                elif mut.dim < range(len(self.layers[layer].stride)):
                    dims = [mut.dim]
                else:
                    dims = []

                if len(dims) == 0:
                    mut.msg = 'Failed to find evolvable stride dimensions'
                    return mut

                for dim in dims:
                    wheel.add((layer, dim), 1)

            while True:

                if wheel.is_empty():
                    mut.msg = 'Empty roulette wheel'
                    return mut

                mut.layer, mut.dim = wheel.pop()

                grow_allowed = self.layers[mut.layer].stride[mut.dim] < self.get_input_shape(mut.layer)[mut.dim + 1] // 2
                shrink_allowed = self.layers[mut.layer].stride[mut.dim] > 1

                if (mut.diff < 0 and not shrink_allowed or
                    mut.diff > 0 and not grow_allowed):
                    continue

                if mut.diff is None:
                    if shrink_allowed:
                        if grow_allowed:
                            mut.diff = -1 if Rand.chance(0.5) else 1
                        else:
                            mut.diff = -1
                    elif grow_allowed:
                        mut.diff = 1

                if mut.diff is not None:
                    break

        # Resize the stride
        print(f'Layer {mut.layer} stride: {self.layers[mut.layer].stride}, diff: {mut.diff}')
        self.layers[mut.layer].stride[mut.dim] += mut.diff

        # Adjust the input size of the following layers
        for layer in range(mut.layer + 1, len(self.layers)):
            self.layers[layer].adjust_input_size(_input_shape = self.get_input_shape(layer))

        mut.success = True
        mut.msg = f'Resized dimension {mut.dim} of stride in layer {mut.layer} by {mut.diff}'

        return mut

    def forward(self,
                _tensor):

        for layer in self.layers:
            _tensor = layer.forward(_tensor)

        return _tensor

    def optimise(self,
                 _data,
                 _target,
                 _optimiser,
                 _loss_function,
                 _output_function,
                 _output_function_args = {}):

        def closure():

            _optimiser.zero_grad()
            loss = _loss_function(_output_function(self(_data), **_output_function_args), _target)
            loss.backward()
            self.fitness.loss_stat.update(loss.item())

        _optimiser.step(closure)

    def crossover(self,
                  _p1,  # Parent 1
                  _p2): # Parent 2

                # Ensure that the nodes in all layers
        # in each parent are up to date
        for layer in _p1.layers:
            layer.update_nodes()
        for layer in _p2.layers:
            layer.update_nodes()

        # Basic sanity check.
        assert len(_p1.layers) > 0, '!!! Error: Parent %r is empty.'.format(_p1.ID)
        assert len(_p2.layers) > 0, '!!! Error: Parent %r is empty.'.format(_p2.ID)

        # First, create reference chromosomes.
        # Roulette wheel for selecting chromosomes
        # (layers) and genes (nodes) from the two
        # parents based on their fitness.
        wheel = Rand.RouletteWheel()
        stat = Stat.SMAStat()
        stat.update(_p1.fitness.relative)
        stat.update(_p2.fitness.relative)

        wheel.add(_p1, stat.get_offset(_p1.fitness.relative))
        wheel.add(_p2, stat.get_offset(_p2.fitness.relative))

#        print(">>> _p1.ID:", _p1.ID)
#        print(">>> _p2.ID:", _p2.ID)

        # First layers of the respective genotypes
        _p1.cur_layer = 0
        _p2.cur_layer = 0

        # Build reference genotypes containing layers
        # collected from the two parents.
        #
        # @note The reference genotypes will end up
        # having the same size even if the two parents
        # have genotypes with different sizes.
        # This is necessary in case speciation is disabled
        # and crossover can happen between any two networks.
        dna1 = []
        dna2 = []

        # Iterate over the layers of the two parents.
        while True:

            if (_p1.cur_layer == len(_p1.layers) and
                _p2.cur_layer == len(_p2.layers)):
                break

            #print(">>> _p1.cur_layer", _p1.cur_layer, " / ", len(_p1.layers))
            #print(">>> _p2.cur_layer", _p2.cur_layer, " / ", len(_p2.layers))

            # Ensure that the types of the reference layers match.
            if (_p1.layers[_p1.cur_layer].role ==
                _p2.layers[_p2.cur_layer].role):

                # Store the parents' layers into the reference genotypes.
                dna1.append(_p1.layers[_p1.cur_layer])
                dna2.append(_p2.layers[_p2.cur_layer])
                _p1.cur_layer += 1
                _p2.cur_layer += 1

            else:

                # Determine the parent with the longer DNA
                if (cl.Layer.Roles[_p1.layers[_p1.cur_layer].role] <
                    cl.Layer.Roles[_p2.layers[_p2.cur_layer].role]):
                    larger_parent = _p1
                    smaller_parent = _p2
                else:
                    larger_parent = _p2
                    smaller_parent = _p1

                # Spin the wheel and lock it into a random position
                wheel.lock()

                while (cl.Layer.Roles[larger_parent.layers[larger_parent.cur_layer].role] <
                       cl.Layer.Roles[smaller_parent.layers[smaller_parent.cur_layer].role]):

                    # If the wheel is locked to the parent with the longer DNA,
                    # store all subsequent layers of the same type from that parent.
                    if wheel.spin().ID == larger_parent.ID:
                        dna1.append(larger_parent.layers[larger_parent.cur_layer])
                        dna2.append(larger_parent.layers[larger_parent.cur_layer])

                        #print(">>> _p1.cur_layer role", _p1.layers[_p1.cur_layer].role)
                        #print(">>> _p2.cur_layer role", _p2.layers[_p2.cur_layer].role)

                    larger_parent.cur_layer += 1

                # Allow the wheel to spin again
                wheel.unlock()

        #=================
        # Crossover
        #=================

        assert len(dna1) == len(dna2), "Crossover failed: DNA length mismatch."

        # Perform crossover for each pair of chromosomes (layers).
        for layer_index in range(len(dna1)):

            wheel.replace([dna1[layer_index], dna2[layer_index]])

            # Compute the shape of the new layer
            shape = [0] * len(list(dna1[layer_index].nodes[0].size()))

#            print(">>> (Crossover for layer", layer_index, ") Shape:", shape)

            # Create a new empty layer
            self.add_layer(_shape = shape,
                           _stride = wheel.spin().stride,
                           _bias = wheel.spin().bias is not None,
                           _layer = layer_index,
                           _activation = wheel.spin().activation)

            # Bias weight values
            bias_weights = []

            # Node counter
            node_index = 0

            # Pick kernels from the two reference chromosomes.
            while (node_index < wheel.elements[0].get_output_nodes() and
                   node_index < wheel.elements[1].get_output_nodes()):

                # Pick a node (gene) from a random layer (chromosome)
                rnd_layer = wheel.spin()
                self.layers[-1].nodes.append(tn.Parameter(rnd_layer.nodes[node_index].clone().detach().requires_grad_(False)))
                self.layers[-1].nodes[-1].requires_grad = self.layers[-1].is_conv

                if rnd_layer.bias is not None:
                    bias_weights.append(rnd_layer.bias[node_index].item())

                node_index += 1

                if (node_index == wheel.elements[0].get_output_nodes() or
                    node_index == wheel.elements[1].get_output_nodes()):

                    # Optionally store any extra nodes.
                    rnd_layer = wheel.spin()
                    wheel.replace([rnd_layer, rnd_layer])

            if self.layers[-1].bias is not None:
                self.layers[-1].bias = tn.Parameter(torch.Tensor(bias_weights))

            self.layers[-1].adjust_input_size()

        # Add the new network to the respective species
        self.species_id = _p1.species_id
        if self.species_id != 0:
            cs.Species.Populations[self.species_id].nets.add(self.ID)

    def clone(self,
              _parent):

        for layer_index, layer in enumerate(_parent.layers):

            _parent.layers[layer_index].update_nodes()

            self.add_layer(_shape = [0, *layer.kernel_size],
                           _stride = layer.stride,
                           _bias = layer.bias is not None,
                           _layer = layer_index,
                           _activation = layer.activation)

            # Clone the nodes
            for node_index, node in enumerate(layer.nodes):
                self.layers[-1].nodes.append(tn.Parameter(node.clone().detach().requires_grad_(False)))
                self.layers[-1].nodes[-1].requires_grad = self.layers[-1].is_conv

            # Clone the bias
            if layer.bias is not None:
                self.layers[-1].bias = tn.Parameter(layer.bias.clone().detach().requires_grad_(False))

            self.layers[-1].update_weights()

        self.species_id = _parent.species_id
        if self.species_id != 0:
            cs.Species.Populations[self.species_id].nets.add(self.ID)

#        assert self.matches(_parent), 'Error cloning network {}'.format(_parent.ID)

    def get_mutation_probabilities(self):

        probabilities = {}

#        # Structural statistics
#        layer_stats = Stat.SMAStat(_title = 'Layers per network')
#        node_stats = Stat.SMAStat(_title = 'Nodes per layer')
#        link_stats = Stat.SMAStat(_title = 'Links per node')
#        stride_stats = Stat.SMAStat(_title = 'Stride')
#        kernel_size_stats = Stat.SMAStat(_title = 'Kernel sizes')
#        kernel_dim_stats = Stat.SMAStat(_title = 'Kernel dimensions')

#        for net in Net.Ecosystem.values():
#            layer_stats.update(len(net.layers))

#        for layer_index, layer in enumerate(self.layers):
#            node_stats.update(layer.get_output_nodes())

#            for node_idx, node in enumerate(layer.nodes):
#                link_stats.update(layer.get_parameter_count(node_idx))

#                if layer.is_conv:

#                    # Kernel size
#                    kernel_size_stats.update(math.pow(Func.prod(layer.kernel_size), 1 / len(layer.kernel_size)))
#                    if len(layer.kernel_size) > kernel_dim_stats.mean:
#                        kernel_dim_stats.update(len(layer.kernel_size))

#                    # Stride
#                    stride_stats.update(Func.prod(layer.get_output_shape()))

#        # Adding or removing a layer involves severing existing links and adding new ones.
#        # For this computation, we assume that the new layer will contain
#        # anywhere between 1 and the mean number of nodes (hence the 0.5).
#        # The SD is a correction term for the average number of nodes
#        # in the following layer whose links would need to be adjusted.
#        probabilities['layer'] = 0.5 * ((node_stats.mean + 1) + node_stats.get_sd()) * link_stats.mean

#        # Adding or removing a node involves adding or removing new links.
#        probabilities['node'] = link_stats.mean

#        # Changing the stride of a layer involves resizing the input of all subsequent layers

#        conv_layer_count = sum([1 for layer in self.layers if layer.is_conv])
#        if conv_layer_count > 0:
#            probabilities['stride'] = 0.5 * stride_stats.mean * link_stats.mean

#        # Growing or shrinking a kernel involves padding the kernel in one of its dimensions.
#        # This is multiplied by the average number of input nodes.
#        kernel_count = sum([len(layer.nodes) for layer in self.layers if layer.is_conv])
#        if kernel_count > 0:
#            probabilities['kernel'] = 2 * node_stats.mean * math.pow(kernel_size_stats.mean, kernel_dim_stats.mean - 1)

        parameter_count = self.get_parameter_count()

        layer_count = len(self.layers)
        probabilities['layer'] = parameter_count / layer_count

        node_count = sum([len(layer.nodes) for layer in self.layers])
        probabilities['node'] = parameter_count / node_count

        conv_layer_count = sum([1 for layer in self.layers if layer.is_conv])
        conv_layer_parameter_count = sum([layer.get_parameter_count() for layer in self.layers if layer.is_conv])

        if conv_layer_count > 0:
            probabilities['stride'] = conv_layer_parameter_count / conv_layer_count

        kernel_count = sum([len(layer.nodes) for layer in self.layers if layer.is_conv])
        if kernel_count > 0:
            probabilities['kernel'] = conv_layer_parameter_count / kernel_count

        return probabilities

    def mutate(self,
               _structure = True,
               _parameters = True,
               _probabilities = None,
               _complexify = None):

        # Statistics about the structure of this network
        probabilities = self.get_mutation_probabilities() if _probabilities is None else _probabilities

#        print('>>> Mutation probabilities:\n{}'.format(probabilities))

        # Complexity can be increased or decreased
        # based on the current complexity of the
        # network relative to the average complexity
        # of the whole population.
#        complexification_chance = (0.5 + self.get_relative_complexity()) / 2
#        complexification_chance = 0.5
        complexification_chance = 0
        complexify = Rand.chance(complexification_chance) if _complexify is None else _complexify

        # The complexity can be increased or decreased
        # with probability proportional to the number
        # of parameters that the mutation will affect.
        wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)

        if _structure:

            wheel.add('layer', probabilities['layer'])

            if len(self.layers) > 1:
                wheel.add('node', probabilities['node'])

            if 'stride' in probabilities:
                wheel.add('stride', probabilities['stride'])

        if _parameters:

            if 'kernel' in probabilities:
                wheel.add('kernel', probabilities['kernel'])

        result = Mutation()

        result.action = 'Complexification' if complexify else 'Simplification'

        if wheel.is_empty():
            result.msg = 'Empty mutation roulette wheel'
            return result

        result.element = wheel.spin()

        for elem_index in range(len(wheel.elements)):
            print(f'{wheel.elements[elem_index]:10s} | {wheel.weights[Rand.WeightType.Raw][elem_index]:10.5f} | {wheel.weights[Rand.WeightType.Inverse][elem_index]:10.5f}')

        #return

        # Do not allow structural mutations if we have reached the limit
        # on the species count and this network's species has more than one member
        if (result.element != 'kernel' and
            cs.Species.Enabled and
            cs.Species.Max.Count > 0 and
            len(cs.Species.Populations) == cs.Species.Max.Count and
            len(cs.Species.Populations[self.species_id].nets) > 1):
            result.msg = 'Species limit reached'
            return result

        # Non-structural mutation
        if result.element == 'kernel':

            mut = self.resize_kernel(_diff = 1) if complexify else self.resize_kernel(_diff = 1)

        elif result.element == 'stride':

            # Growing the stride actually reduces the number of parameters
            mut = self.resize_stride(_diff = -1) if complexify else self.resize_stride(_diff = 1)

        elif result.element == 'layer':

            mut = self.add_layer() if complexify else self.remove_layer()

        elif result.element == 'node':

            mut = self.add_nodes() if complexify else self.remove_nodes()

        mut.action = result.action
        mut.element = result.element

        if not mut.success:
            print(f'[Net {self.ID}] >>> {mut.action} failed: {mut.msg}')

        if (mut.success and    # If the mutation was successful
            mut.element != 'kernel' and  # ...and non-strucutral
            cs.Species.Enabled and   # ...and speciation is enabled
            self.species_id > 0):    # ...and the network is not isolated

            # Create a new species
            new_species_id = cs.Species.find(_genome = self.get_genome())
            if new_species_id == 0:
                new_species = cs.Species(_genome = self.get_genome())
            else:
                new_species = cs.Species.Populations[new_species_id]

            # Remove the network from the current species
            print('\t>>> Removing net {} from species {}'.format(self.ID, self.species_id))
            cs.Species.Populations[self.species_id].nets.remove(self.ID)

            # Add the network to the new species
            print('\t>>> Adding net {} to species {}'.format(self.ID, new_species.ID))
            new_species.nets.add(self.ID)

            # Remove the species from the ecosystem if it has gone extinct
            if len(cs.Species.Populations[self.species_id].nets) == 0:
                print(f'>>> Removing extinct species {self.species_id}')
                del cs.Species.Populations[self.species_id]

            # Store the species ID in this network
            self.species_id = new_species.ID

        if mut.success:
            print(f'[Net {self.ID}] >>> {mut.action} successful: {mut.msg}')

        return mut
