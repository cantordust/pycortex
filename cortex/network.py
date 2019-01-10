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
                    print('>>> Performing crossover between networks {} and {}'.format(_p1.ID, _p2.ID))
                    self.crossover(_p1, _p2)
                else:
                    print('>>> Cloning network {}'.format(_p1.ID))
                    self.clone(_p1)

            else:

                layer_defs = _species.genome if isinstance(_species, cs.Species) else Net.Init.Layers

                if len(layer_defs) > 0:
                    for layer_index, layer_def in enumerate(layer_defs):
                        self.add_layer(_shape = layer_def.shape,
                                       _bias = layer_def.bias,
                                       _activation = layer_def.activation,
                                       _layer_index = layer_index)

                # Output layer
                self.add_layer(_shape = Net.Output.Shape,
                              _bias = cl.Layer.Bias,
                              _layer_index = len(self.layers))

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

    def print(self,
              _file = sys.stdout):

        print("\n###################[ Network", self.ID, "]###################\n",
              "\n>>> Fitness:",
              "\n\tAbsolute:", self.fitness.absolute,
              "\n\tRelative:", self.fitness.relative,
              "\n\n>>> Age:", self.age,
              "\n>>> Species:", self.species_id,
              "\n>>> Total parameters:", self.get_parameter_count(),
              file = _file)

        for layer in self.layers:
            layer.print(_file)

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
#                                       _stride = [0] * len(layer.stride),
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
                        _layer_index):
        """
        Compute the shape of this layer's input.
        """

        if _layer_index <= 0:
            return Net.Input.Shape

        assert _layer_index <= len(self.layers), "(get_input_shape) Invalid layer index %r (network %r contains %r layer(s))" % (_layer_index, self.ID, len(self.layers))

        return self.layers[_layer_index - 1].get_output_shape()

    def get_output_shape(self,
                         _layer_index):
        """
        Compute this layer's output shape based on the shape of the preceding layer
        (or the input shape if it is the first layer) and the layer's kernels.
        """

        assert _layer_index < len(self.layers), "(get_output_shape) Invalid layer index %r (network %r contains %r layers)" % (_layer_index, self.ID, len(self.layers))

        return self.layers[_layer_index].get_output_shape()

    def get_allowed_layer_shapes(self,
                                 _layer_index):

        """
        Check what layer shapes are allowed at this index
        """
        # The output layer has a predetermined shape
        if _layer_index >= len(self.layers):
            return [Net.Output.Shape]

        input_shape = [0] * len(self.get_input_shape(_layer_index))
        output_shape = [0] * len(self.get_output_shape(_layer_index))

        return [input_shape] if len(input_shape) == len(output_shape) else [input_shape, output_shape]

    def get_parameter_count(self):

        parameters = sum([param.numel() for param in self.parameters() if param.requires_grad])
#        print("Net", self.ID, "parameter count:", parameters)

        return parameters

    def get_complexity(self):

        complexity_stat = Stat.SMAStat()
        for net in Net.Ecosystem.values():
            complexity_stat.update(net.get_parameter_count())

        return complexity_stat.get_offset(net.get_parameter_count())

    def add_layer(self,
                  _shape = [],
                  _stride = [],
                  _bias = None,
                  _activation = None,
                  _layer_index = None,
                  _test = False):

        if (_layer_index is None or
            len(_shape) == 0):

            wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)

            for layer_index in range(len(self.layers)):

                # Check how many links we have to add and / or remove
                # to insert a layer of each allowed shape

                node_stat = Stat.SMAStat()
                for layer in self.layers:
                    node_stat.update(len(layer.nodes))

#                new_nodes = math.floor(_stats['nodes'].mean)
                for allowed_shape in self.get_allowed_layer_shapes(layer_index):

                    if (len(_shape) > 0 and
                        _shape[0] > 0):
                        new_nodes = _shape[0]
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
                return (False, _layer_index)

            layer_index, shape = wheel.spin()

            if _layer_index is None:
                _layer_index = layer_index

            if len(_shape) == 0:
                _shape = shape

        # Sanity check for the layer index
        if _layer_index > len(self.layers):
            #print("Invalid layer index", _layer_index)
            return (False, _layer_index)

        # Create a layer definition if not provided
        layer_def = cl.Layer.Def(_shape, _stride, _bias, _activation)

        input_shape = self.get_input_shape(_layer_index)

        # Ensure that the layer roles are contiguous.
        # This can be done with a simple comparison of the input and output shapes.
        if (len(input_shape) < len(layer_def.shape) or                              # Attempting to add a conv layer above an FC one
            (_layer_index < len(self.layers) and
             len(layer_def.shape) < len(self.get_output_shape(_layer_index)))):     # Attempting to add an FC layer below a conv one

            print('[Net {}] >>> Invalid layer size: Cannot add layer of shape {} at position {}'.format(self.ID, layer_def.shape, _layer_index))

            return (False, _layer_index)

        # Create the new layer
        new_layer = cl.Layer(_layer_def = layer_def,
                             _input_shape = input_shape,
                             _layer_index = _layer_index)

        print('[Net {}] >>> Adding new {} layer with shape {} at position {}'.format(self.ID, new_layer.role, layer_def.shape, _layer_index))

        # Rearrange the module stack if necessary
        if _layer_index == len(self.layers):
            self.layers.append(new_layer)

        else:
            new_ml = tn.ModuleList()

            for index in range(len(self.layers)):
                if index == _layer_index:
                    new_ml.append(new_layer)

                new_ml.append(self.layers[index])

            self.layers = new_ml

        # Adjust the input size of the following layers
        for layer_index in range(_layer_index + 1, len(self.layers)):
            self.layers[layer_index].adjust_input_size(_input_shape = self.get_input_shape(layer_index))

        self.reindex()

        return (True, _layer_index)

    def remove_layer(self,
                    _layer_index = None):

        if len(self.layers) < 2:
            return (False, _layer_index)

        # Update the input size of the next layer (if there is one)
        if _layer_index is None:

            wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)
            for layer_index in range(len(self.layers) - 1):
                wheel.add((layer_index,), 1)

            if wheel.is_empty():
                return (False, _layer_index)

            _layer_index = wheel.spin()[0]

        # We cannot remove the output layer
        if _layer_index == len(self.layers) - 1:
            return (False, _layer_index)

        print('[Net {}] >>> Removing {} layer at position {}'.format(self.ID, self.layers[_layer_index].role, _layer_index))

        # Remove the layer
        del self.layers[_layer_index]

        # Adjust the input size of the following layers
        for layer_index in range(_layer_index, len(self.layers)):
            self.layers[layer_index].adjust_input_size(_input_shape = self.get_input_shape(layer_index))

        self.reindex()

        return (True, _layer_index)

    def add_nodes(self,
                  _layer_index = None,
                  _count = 1,
                  _max_radius = []):

        if _count <= 0:
            return (False, _layer_index, set())

        if _layer_index is None:

            wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)
            for layer_index in range(len(self.layers) - 1):
                wheel.add((layer_index,), 1)

            if wheel.is_empty():
                return (False, _layer_index, set())

            _layer_index = wheel.spin()[0]

        node_indices = set()
        for n in range(len(self.layers[_layer_index].nodes), len(self.layers[_layer_index].nodes) + _count):
            node_indices.add(n)

        # Sanity checks
        if (len(self.layers) == 0 or
            _layer_index == len(self.layers) - 1):
                return (False, _layer_index, node_indices)

        print('[Net {}] >>> Adding {} node(s) to layer {}'.format(self.ID, len(node_indices), _layer_index))

        # Add the nodes
        success = self.layers[_layer_index].add_nodes(_count, _max_radius)

        # On success, adjust the input size of the next layer if there is one
        if (success and
            _layer_index < len(self.layers) - 1):
            self.layers[_layer_index + 1].adjust_input_size(_input_shape = self.get_output_shape(_layer_index))

        return (success, _layer_index, node_indices)

    def remove_nodes(self,
                    _layer_index = None,
                    _count = 1,
                    _node_indices = set()):

        # Sanity checks
        if isinstance(_node_indices, int):
            node_index = _node_indices
            _node_indices = set()
            _node_indices.add(node_index)

        elif isinstance(_node_indices, list):
            node_indices = set()
            for node_index in _node_indices:
                node_indices.add(node_index)
            _node_indices = node_indices

        if _count <= 0:
            return (False, _layer_index, _node_indices)

        if (_layer_index is None or
            len(_node_indices) == 0):

            wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)

            for layer_index in range(len(self.layers) - 1):

                # Check if we can remove a node at all
                if len(self.layers[layer_index].nodes) > 1:

                    for node_index in range(len(self.layers[layer_index].nodes)):
                        wheel.add((layer_index, node_index), 1)

            if wheel.is_empty():
                return (False, _layer_index, _node_indices)

            layer_index, node_index = wheel.spin()

            # Create a new wheel only for the nodes in the selected layer
            new_wheel = Rand.RouletteWheel()
            for index in range(len(wheel.elements)):
                if wheel.elements[index][0] == layer_index:
                    new_wheel.add(wheel.elements[index][1], wheel.weights[Rand.WeightType.Raw][index])

            wheel = new_wheel

            # Store the layer index
            if _layer_index is None:
                _layer_index = layer_index

            # Populate the node indices
            if len(_node_indices) == 0:
                _node_indices = set()
                # Check if we have enough nodes to work with
                if _count >= len(self.layers[_layer_index].nodes):
                    return (False, _layer_index, _node_indices)

                # Draw the necessary number of node indices
                while len(_node_indices) < _count:
                    _node_indices.add(wheel.pop())

        # Sanity check for the layer index
        if (len(self.layers) == 0 or
            _layer_index == len(self.layers) - 1):
                return (False, _layer_index, _node_indices)

        # Sanity check for node indices.
        if len(_node_indices) > 0:
            for node_index in _node_indices:
                if (node_index < 0 or
                    node_index >= len(self.layers[_layer_index].nodes)):
                    return (False, _layer_index, _node_indices)
                _count = -len(_node_indices)

        print('[Net {}] >>> Removing node(s) {} from layer {}'.format(self.ID, *_node_indices, _layer_index))

        # Remove the nodes
        success = self.layers[_layer_index].remove_nodes(sorted(list(_node_indices)))

        # On success, adjust the input size of the next layer if there is one
        if (success and
            _layer_index < len(self.layers) - 1):
            self.layers[_layer_index + 1].adjust_input_size(_input_shape = self.get_output_shape(_layer_index),
                                                            _node_indices = sorted(list(_node_indices)))

        return (success, _layer_index, _node_indices)

    def grow_kernel(self,
                    _layer_index = None,
                    _node_index = None,
                    _delta = {}):

        if len(_delta) > 1:
            return (False, _layer_index, _node_index, _delta)

        if (_layer_index is None or
            _node_index is None or
            len(_delta) == 0):

            wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)

            for layer_index, layer in enumerate(self.layers):

                # For now, this only operates on kernels in convolutional layers
                if layer.is_conv:

                    for node_index, node in enumerate(layer.nodes):

                        # Check how many links we would have to add or remove
                        for dim in range(len(layer.kernel_size)):
                            wheel.add((layer_index, node_index, dim), 1)

            if wheel.is_empty():
                return (False, _layer_index, _node_index, _delta)

            layer_index, node_index, dim = wheel.spin()

            if _layer_index is None:
                _layer_index = layer_index

            if _node_index is None:
                _node_index = node_index

            if len(_delta) == 0:
                _delta = {dim: 1}

        # Ensure all deltas are positive
        for dim in _delta.keys():
            _delta[dim] = abs(_delta[dim])

        print('[Net {}] >>> Growing dimension {} of kernel {} in layer {} by {}'.format(self.ID, *_delta.keys(), _node_index, _layer_index, *_delta.values() ))

        # Grow the kernel
        success = self.layers[_layer_index].resize_kernel(_node_index, _delta)

        return (success, _layer_index, _node_index, _delta)

    def shrink_kernel(self,
                      _layer_index = None,
                      _node_index = None,
                      _delta = {}):

        if len(_delta) > 1:
            return (False, _layer_index, _node_index, _delta)

        if (_layer_index is None or
            _node_index is None or
            len(_delta) == 0):

            wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)

            for layer_index, layer in enumerate(self.layers):

                # For now, this only operates on kernels in convolutional layers
                if layer.is_conv:

                    for node_index, node in enumerate(layer.nodes):

                        # Check how many links we would have to add or remove
                        for dim in range(len(layer.kernel_size)):

                            if node.size(dim + 1) > 1:
                                wheel.add((layer_index, node_index, dim), 1)

            if wheel.is_empty():
                return (False, _layer_index, _node_index, _delta)

            layer_index, node_index, dim = wheel.spin()

            if _layer_index is None:
                _layer_index = layer_index

            if _node_index is None:
                _node_index = node_index

            if len(_delta) == 0:
                _delta = {dim: -1}

        # Ensure all deltas are negative
        for dim in _delta.keys():
            _delta[dim] = -abs(_delta[dim])

        print('[Net {}] >>> Shrinking dimension(s) {} of kernel {} in layer {} by {}'.format(self.ID, *_delta.keys(), _node_index, _layer_index, abs(*_delta.values()) ))

        # Shrink the kernel
        success = self.layers[_layer_index].resize_kernel(_node_index, _delta)

        return (success, _layer_index, _node_index, _delta)

    def grow_stride(self,
                    _layer_index = None,
                    _delta = {}):

        if _layer_index is None:
            wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)

            for layer_index, layer in enumerate(self.layers):

                if not layer.is_conv:
                    break

                for dim in range(len(layer.stride)):
                    if layer.stride[dim] < self.get_input_shape(layer_index)[dim + 1] // 2:
                        wheel.add((layer_index, dim), 1)

            if wheel.is_empty():
                return (False, _layer_index, _delta)

            layer_index, dim = wheel.spin()

            if _layer_index is None:
                _layer_index = layer_index

            if len(_delta) == 0:
                _delta = {dim: 1}

        # Ensure all deltas are positive
        for dim in _delta.keys():
            _delta[dim] = abs(_delta[dim])

        print('[Net {}] >>> Growing dimension {} of stride in layer {} by {}'.format(self.ID, *_delta.keys(), _layer_index, *_delta.values() ))

        # Grow the stride
        for dim in _delta.keys():
            self.layers[_layer_index].stride[dim] += _delta[dim]

        # Adjust the input size of the following layers
        for layer_index in range(_layer_index + 1, len(self.layers)):
            self.layers[layer_index].adjust_input_size(_input_shape = self.get_input_shape(layer_index))

        return (True, _layer_index, _delta)

    def shrink_stride(self,
                      _layer_index = None,
                      _delta = {}):

        if _layer_index is None:
          wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)

          for layer_index, layer in enumerate(self.layers):

              # Check how many links we have to add and / or remove
              # to insert a layer of each allowed shape
              if not layer.is_conv:
                  break

              for dim in range(len(layer.stride)):
                  if layer.stride[dim] > 1:
                      wheel.add((layer_index, dim), 1)

          if wheel.is_empty():
              return (False, _layer_index, _delta)

          layer_index, dim = wheel.spin()

          if _layer_index is None:
              _layer_index = layer_index

          if len(_delta) == 0:
              _delta = {dim: -1}

        # Ensure all deltas are negative
        for dim in _delta.keys():
            _delta[dim] = -abs(_delta[dim])

        print('[Net {}] >>> Shrinking dimension {} of stride in layer {} by {}'.format(self.ID, *_delta.keys(), _layer_index, abs(*_delta.values()) ))

        # Shrink the stride
        for dim in _delta.keys():
            # Delta is negative, so + means shrink
            self.layers[_layer_index].stride[dim] += _delta[dim]

        # Adjust the input size of the following layers
        for layer_index in range(_layer_index + 1, len(self.layers)):
            self.layers[layer_index].adjust_input_size(_input_shape = self.get_input_shape(layer_index))

        return (True, _layer_index, _delta)

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
                           _layer_index = layer_index,
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

            layer.update_nodes()

            self.add_layer(_shape = [0, *layer.kernel_size],
                           _stride = layer.stride,
                           _bias = layer.bias is not None,
                           _layer_index = layer_index,
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
        complexify = Rand.chance((0.5 + self.get_complexity()) / 2) if _complexify is None else _complexify

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

        if wheel.is_empty():
            return False

        element = wheel.spin()

        print("Mutating network", self.ID)

        for elem_index in range(len(wheel.elements)):
            print(wheel.elements[elem_index], "|\t", wheel.weights[Rand.WeightType.Raw][elem_index], "|\t", wheel.weights[Rand.WeightType.Inverse][elem_index])

        #return

        # Non-structural mutation
        if element == 'kernel':

            print('\t>>> {} {}'.format('Growing' if complexify else 'Shrinking', element))
            success, layer, node, delta = self.grow_kernel() if complexify else self.shrink_kernel()

        # Structural mutation
        else:

            # Do not allow structural mutations if we have reached the limit
            # on the species count and this network's species has more than one member
            if (cs.Species.Enabled and
                cs.Species.Max.Count > 0 and
                len(cs.Species.Populations) == cs.Species.Max.Count and
                len(cs.Species.Populations[self.species_id].nets) > 1):
                return False

            if element == 'layer':

                print('\t>>> {} {}'.format('Adding' if complexify else 'Removing', element))
                success, layer = self.add_layer() if complexify else self.remove_layer()

            elif element == 'node':

                print('\t>>> {} {}'.format('Adding' if complexify else 'Removing', element))
                success, layer, nodes = self.add_nodes() if complexify else self.remove_nodes()

            elif element == 'stride':

                # Growing the stride actually reduces the number of parameters
                print('\t>>> {} {}'.format('Shrinking' if complexify else 'Growing', element))
                success, layer, delta = self.shrink_stride() if complexify else self.grow_stride()

            if (success and              # If the mutation was successful
                cs.Species.Enabled and   # ...and speciation is enabled
                self.species_id > 0):    # ...and the network is not isolated

                # Create a new species
                new_species_id = cs.Species.find(_genome = self.get_genome())
                if new_species_id == 0:
                    new_species = cs.Species(_genome = self.get_genome())
                else:
                    new_species = cs.Species.Populations[new_species_id]

                # Add the network to the new species
                print('\t>>> Adding net {} to species {}'.format(self.ID, new_species.ID))
                new_species.nets.add(self.ID)

                # Remove the network from the current species
                print('\t>>> Removing net {} from species {}'.format(self.ID, self.species_id))
                cs.Species.Populations[self.species_id].nets.remove(self.ID)

                # Remove the species from the ecosystem if it has gone extinct
                if len(cs.Species.Populations[self.species_id].nets) == 0:
                    print('\t>>> Removing extinct species {}'.format(self.species_id))
                    del cs.Species.Populations[self.species_id]

                # Store the species ID in this network
                self.species_id = new_species.ID

        return success
