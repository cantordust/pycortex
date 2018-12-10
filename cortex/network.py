# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 22:49:52 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

#from __future__ import print_function

import sys
import math

import torch
import torch.nn as tn
import torch.nn.functional as tnf

from .species import Species
from .layer import Layer
from .random import WeightType, RouletteWheel
from . import random as Rand
from . import functions as Func

torch.set_printoptions(precision = 4, threshold = 5000, edgeitems = 5, linewidth = 160)

class Net(tn.Module):

    class Input:
        Shape = [1, 28, 28]

    class Output:
        Shape = [10]
        Bias = None
        Activation = None

    class Init:
        Count = 50
        Layers = [Layer.Def([10, 0, 0]),
                  Layer.Def([10, 0, 0]),
                  Layer.Def(10)]
        Func = tn.init.uniform_
        Args = {'a': -0.05, 'b': 0.05}

    class Max:
        Count = 200
        Age = 50

    champ = None
    population = {}
    ID = 0
    Loss = tnf.cross_entropy

    @staticmethod
    def get_structure_stats(_net_id = None):

        from . import statistics as Stat

        net_list = Net.population.keys() if _net_id is None else [_net_id]

        # Structural statistics
        layer_stats = Stat.SMAStat(_title = "Layers per network")
        node_stats = Stat.SMAStat(_title = "Nodes per layer")
        link_stats = Stat.SMAStat(_title = "Links per node")
        kernel_size_stats = Stat.SMAStat(_title = "Kernel sizes")
        kernel_dims = len(Net.Input.Shape) - 1

        for net_id in net_list:
            net = Net.population[net_id]

            # Add the input nodes to the count
            node_stats.update(Net.Input.Shape[0])

            layer_stats.update(len(net.layers))
            for layer in net.layers:
                node_stats.update(layer.get_output_nodes())
                for node_idx in range(len(layer.nodes)):
                    link_stats.update(layer.get_link_count(node_idx))
                    if layer.is_conv:
                        kernel_size_stats.update(math.pow(Func.prod(layer.kernel_size), 1 / len(layer.kernel_size)))
                        if len(layer.kernel_size) > kernel_dims:
                            kernel_dims = len(layer.kernel_size)

        return {'layers': layer_stats,
                'nodes': node_stats,
                'links': link_stats,
                'kernel_sizes': kernel_size_stats,
                'kernel_dims': kernel_dims}

    def get_parametric_complexity(self):

        from . import statistics as Stat

        global_parameter_count = Stat.SMAStat()
        self_parameter_count = 0

        for ID, net in Net.population.items():

            parameters = 0
            for layer in self.layers:
                parameters += layer.get_link_count()

            global_parameter_count.update(parameters)

            if ID == self.ID:
                self_parameter_count = parameters

        return global_parameter_count.get_offset(self_parameter_count)

    def get_structural_complexity(self):
        
        from . import statistics as Stat

        global_node_count = Stat.SMAStat()
        self_node_count = 0

        for ID, net in Net.population.items():

            nodes = 0
            for layer in self.layers:
                nodes += layer.get_output_nodes()

            global_node_count.update(nodes)

            if ID == self.ID:
                self_node_count = nodes

        return global_node_count.get_offset(self_node_count)

    def __init__(self,
                 _empty = False,
                 _p1 = None,
                 _p2 = None,
                 _species = None,
                 _isolated = False):

        super(Net, self).__init__()

        from .species import Species
        from .fitness import Fitness

        # Assign a network ID
        self.ID = 0
        
        if not _isolated:
            # Increment the ID counter
            Net.ID += 1
            self.ID = Net.ID

            # Add this network to the population
            Net.population[self.ID] = self

        # Initialise the age
        self.age = 0

        # Initialise the age
        self.fitness = Fitness()

        # Generate modules
        self.layers = tn.ModuleList()

        # Species
        self.species_id = 0
        
        if not _isolated:
            
            if isinstance(_species, Species):
                self.species_id = _species.ID
                Species.env[self.species_id].nets.add(self.ID)

        if not _empty:

            if (isinstance(_p1, Net) and 
                isinstance(_p2, Net)):
                self.crossover(_p1, _p2)

            else:

                layer_defs = _species.genome if isinstance(_species, Species) else Net.Init.Layers

                if len(layer_defs) > 0:
                    for layer_index, layer_def in enumerate(layer_defs):
                        self.add_layer(_shape = layer_def.shape, 
                                       _bias = layer_def.bias,
                                       _activation = layer_def.activation,
                                       _layer_index = layer_index)

                # Output layer
                self.add_layer(_shape = Net.Output.Shape,
                               _bias = Net.Output.Bias,
                               _activation = Net.Output.Activation,
                               _layer_index = len(self.layers))
                    
#        print(">>> Network", self.ID, "created")

    def matches(self,
                _other):
        
        if len(self.layers) != len(_other.layers):
            print("\t>>> Different number of layers")
            print("\t>>> Network 1:\n")
            self.print()
            print("\t>>> Network 2:\n")
            _other.print()
            return False
        
        for layer_index in range(len(self.layers)):
            if not self.layers[layer_index].matches(_other.layers[layer_index]):
                return False
        
        return True

    def print(self,
              _file = None,
              _truncate = False):
        
        if _file is None:
            fh = sys.stdout
        if isinstance(_file, str):
            fh = open(_file, 'w')
            if _truncate:
                fh.truncate()
        else:
            fh = _file

        print("\n###################[ Network", self.ID, "]###################\n", file = fh)
        for layer in self.layers:
            layer.print(fh)
            
    def is_conv(self):
        # To determine whether this network is convolutional,
        # it is enough to check if the first layer is convolutional.
        if len(self.lyaers) > 0:
            return self.layers[0].is_conv
        return False

    def get_genome(self):

        genome = []
        for layer in self.layers:
            genome.append(Layer.Def(_shape = layer.get_output_shape(),
                                    _bias = layer.bias is not None,
                                    _activation = layer.activation,
                                    _type = layer.type))
        return genome

    def reindex(self):
        for index, layer in enumerate(self.layers):
            layer.index = index
            
            if index == len(self.layers) - 1:
                layer.type = 'output'
            else:
                layer.type = layer.op.__name__ 

    def get_input_shape(self,
                        _layer_index):
        """
        Compute the shape of this layer's input.
        """

        if _layer_index <= 0:
            return list(Net.Input.Shape)

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

    def add_layer(self,
                  _shape = [],
                  _bias = None,
                  _activation = None,
                  _layer_index = None,
                  _stats = None,
                  _nodes = None,
                  _bias_nodes = None,
                  _test = False):

        if (_layer_index is None or
            len(_shape) == 0):

            if _stats is None:
                _stats = self.get_structure_stats(self.ID)

            wheel = RouletteWheel()

            for layer_index in range(len(self.layers)):

                # Check how many links we have to add and / or erase
                # to insert a layer of each allowed shape
                for shape in self.get_allowed_layer_shapes(layer_index):
                    
                    link_count = []
                    new_layer_shape = list(shape)

                    # Set the number of output nodes
                    if _test:
                        new_layer_shape = [0] * len(self.get_input_shape(layer_index))
                        new_layer_shape[0] = self.layers[layer_index].get_input_nodes() + 1
                        
                    elif new_layer_shape[0] == 0:
                        new_layer_shape[0] = math.floor(_stats['nodes'].mean)
                   
                    # Set the kernel size to 1 if the shape corresponds to a convolutional layer
                    for dim in range(1,len(new_layer_shape)):
                        new_layer_shape[dim] = 1
                   
                    # Compute the output shape of a hypothetical layer of this shape
                    new_output_shape = Layer.compute_output_shape(new_layer_shape[0],
                                                                  self.get_input_shape(layer_index),
                                                                  new_layer_shape[1:])
                    
                    #print("Input shape:", self.get_input_shape(layer_index))
                    #print("New output shape:", new_output_shape)
                   
                    #print("Number of links affected by adding layer with shape", new_layer_shape, "at index", layer_index)

                    # Links to add
                    input_shape = self.get_input_shape(layer_index)

                    # The shape corresponds to an FC layer.
                    link_count.append(new_layer_shape[0] * Func.prod(input_shape[0:len(input_shape) - len(new_output_shape) + 1]))

                    #print("\t>>> Add:", link_count[-1])

                    # Links to adjust
                    if layer_index < len(self.layers):
                        # Only compute the number of links to erase if
                        # the new layer is *not* going to be the output layer.
                        link_count.append(self.layers[layer_index].adjust_input_size(_input_shape = new_output_shape, 
                                                                                     _pretend = True))

                        #print("\t>>> Adjust:", link_count[-1])

                    #print("\t>>> Total:", sum(link_count))

                    wheel.add((layer_index, new_layer_shape), 1 / sum(link_count))

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
        layer_def = Layer.Def(_shape, _bias, _activation)

        input_shape = self.get_input_shape(_layer_index)

        # Ensure that the layer types are contiguous.
        # This can be done with a simple comparison of the input and output shapes.
        if (len(input_shape) < len(layer_def.shape) or                              # Attempting to add a conv layer above an FC one
            (_layer_index < len(self.layers) and
             len(layer_def.shape) < len(self.get_output_shape(_layer_index)))):     # Attempting to add an FC layer below a conv one
            
            print(">>> Invalid layer size: Cannot add layer of shape %r at position %r" % (layer_def.shape, _layer_index))
            
            return (False, _layer_index)

        # Create the new layer
        new_layer = Layer(_layer_def = layer_def,
                          _input_shape = input_shape,
                          _layer_index = _layer_index,
                          _nodes = _nodes,
                          _bias_nodes = _bias_nodes)
                
        print(">>> Adding new", new_layer.type , "layer with shape", _shape, "at position", _layer_index)

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

        # Adjust the input size of the following layer
        if _layer_index < len(self.layers) - 1:
            self.layers[_layer_index + 1].adjust_input_size(_input_shape = new_layer.get_output_shape())

        self.reindex()

        return (True, _layer_index)

    def erase_layer(self,
                    _layer_index = None):

        if len(self.layers) < 2:
            return (False, _layer_index)

        # Update the input size of the next layer (if there is one)
        if _layer_index is None:

            wheel = RouletteWheel()
            for layer_index in range(len(self.layers) - 1):

                # Check if we can erase a layer at all.
                link_count = []
                
                #print("Number of links affected by erasing layer", layer_index)

                # Links to erase
                link_count.append(self.layers[layer_index].get_link_count())

                #print("\t>>> Erase:", link_count[-1])

                # Links to adjust
                link_count.append(self.layers[layer_index + 1].adjust_input_size(_input_shape = self.get_input_shape(layer_index), 
                                                                                 _pretend = True))

                #print("\t>>> Adjust:", link_count[-1])

                #print("\t>>> Total:", sum(link_count))

                wheel.add((layer_index,), 1 / sum(link_count))

            if wheel.is_empty():
                return (False, _layer_index)

            _layer_index = wheel.spin()[0]

        # We cannot erase the output layer
        if _layer_index == len(self.layers) - 1:
            return (False, _layer_index)

        # Adjust the input size of the following layer
        self.layers[_layer_index + 1].adjust_input_size(_input_shape = self.get_input_shape(_layer_index))

        print(">>> Erasing", self.layers[_layer_index].type , "layer at position", _layer_index)
       
        # Remove the layer
        del self.layers[_layer_index]

        self.reindex()

        return (True, _layer_index)

    def add_nodes(self,
                  _layer_index = None,
                  _count = 1,
                  _max_radius = [],
                  _stats = None):

        if _count <= 0:
            return (success, _layer_index, set())

        if _layer_index is None:

            if _stats is None:
                _stats = self.get_structure_stats(self.ID)

            wheel = RouletteWheel()
            for layer_index in range(len(self.layers) - 1):

                layer = self.layers[layer_index]
                output_shape = layer.get_output_shape()
                
                link_count = []
                
                #print("Number of links affected by adding a node to layer", layer_index)

                # Check how many links we would have to add
                link_count.append(layer.get_mean_link_count()) # * 1 node
                
                #print("\t>>> Add:", link_count[-1])
                
                # Adjust the input size of the next layer
                new_output_shape = list(output_shape)
                new_output_shape[0] += 1
                link_count.append(self.layers[layer_index + 1].adjust_input_size(_input_shape = new_output_shape, 
                                                                                 _pretend = True))
                
                #print("\t>>> Adjust:", link_count[-1])
                
                #print("\t>>> Total:", sum(link_count))
                
                wheel.add((layer_index,), 1 / sum(link_count))

            if wheel.is_empty():
                return (False, _layer_index, set())

            _layer_index = wheel.spin()[0]

        node_indices = set()
        for n in range(len(self.layers[_layer_index].nodes), len(self.layers[_layer_index].nodes) + _count):
            node_indices.add(n)

        # Sanity checks
        if (len(self.layers) == 0 or
            _layer_index == len(self.layers) - 1 or
            (_layer_index == 0 and
            not self.layers[_layer_index].is_conv)):
                return (False, _layer_index, node_indices)

        print(">>> Adding", _count, "node" if _count == 1 else "nodes", "to layer", _layer_index)

        # Add the nodes
        success = self.layers[_layer_index].add_nodes(_count, _max_radius)
        
        # On success, adjust the input size of the next layer if there is one
        if (success and
            _layer_index < len(self.layers) - 1):
            self.layers[_layer_index + 1].adjust_input_size(_input_shape = self.get_output_shape(_layer_index))
            
        return (success, _layer_index, node_indices)

    def erase_nodes(self,
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

        if (_layer_index is None or 
            len(_node_indices) == 0):

            wheel = RouletteWheel()

            for layer_index in range(len(self.layers) - 1):
                # Check if we can erase a node at all:
                # Check how many links we would have to erase

                layer = self.layers[layer_index]
                
                if len(layer.nodes) > 1:

                    layer_shape = layer.get_output_shape()

                    for node_index in range(len(layer.nodes)):

                        link_count = []
                        
                        #print("Number of links affected by erasing node", node_index, "from layer", layer_index)

                        # Check how many links we would have to erase
                        link_count.append(layer.nodes[node_index].numel())
                        
                        #print("\t>>> Erase:", link_count[-1])
                        
                        # Add the number of links lost through adjusting
                        # the size of the next layer
                        new_layer_shape = list(layer_shape)
                        new_layer_shape[0] -= 1
                        link_count.append(self.layers[layer_index + 1].adjust_input_size(_input_shape = new_layer_shape, 
                                                                                         _pretend = True))
                        
                        #print("\t>>> Adjust:", link_count[-1])

                        #print("\t>>> Total:", sum(link_count))

                        wheel.add((layer_index, node_index), 1 / sum(link_count))

            if wheel.is_empty():                
                return (False, _layer_index, _node_indices)

            layer_index, node_index = wheel.spin()
            
            # Create a new wheel only for the nodes in the selected layer
            new_wheel = RouletteWheel()
            for index in range(len(wheel.elements)):
                if wheel.elements[index][0] == layer_index:
                    new_wheel.add(wheel.elements[index][1], wheel.weights[WeightType.Raw][index])
                    
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
            _layer_index == len(self.layers) - 1 or
            (_layer_index == 0 and
            not self.layers[_layer_index].is_conv)):
                return (False, _layer_index, _node_indices)
            
        # Sanity check for node indices.
        if len(_node_indices) > 0:
            for node_index in _node_indices:
                if not(0 <= node_index < len(self.layers[_layer_index].nodes)):
                    return (False, _layer_index, _node_indices)
                _count = -len(_node_indices)

        print(">>> Erasing node(s)", *_node_indices, "from layer", _layer_index)
        
        # Erase the nodes
        success = self.layers[_layer_index].erase_nodes(sorted(list(_node_indices)))
        
        # On success, adjust the input size of the next layer if there is one
        if (success and
            _layer_index < len(self.layers) - 1):
            self.layers[_layer_index + 1].adjust_input_size(_input_shape = self.get_output_shape(_layer_index),
                                                            _node_indices = sorted(list(_node_indices)))
        
        return (success, _layer_index, _node_indices)

    def grow_kernel(self,
                    _layer_index = None,
                    _node_index = None,
                    _delta = {},
                    _stats = None):

        if len(_delta) > 1:
            return (False, _layer_index, _node_index, _delta)

        if (_layer_index is None or
            _node_index is None or
            len(_delta) == 0):

            if _stats is None:
                _stats = self.get_structure_stats(self.ID)

            wheel = RouletteWheel()

            for layer_index, layer in enumerate(self.layers):

                # For now, this only operates on kernels in convolutional layers
                if layer.is_conv:

                    for node_index, node in enumerate(layer.nodes):

                        # Check how many links we would have to add or erase
                        for dim in range(len(layer.kernel_size)):

                            #print("Number of links affected by growing dimension", dim, "of kernel", node_index, "in layer", layer_index)
                            
                            link_count = 2 * layer.get_input_nodes() * math.pow(node.size(dim + 1), len(layer.kernel_size) - 1)

                            #print("\t>>> Total:", link_count)

                            # Check how many links we would have to add
                            wheel.add((layer_index, node_index, dim), 1 / link_count)

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
            
        print(">>> Growing dimension", *_delta.keys(), "of node", _node_index, "in layer", _layer_index, "by", *_delta.values() )

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

            wheel = RouletteWheel()

            for layer_index, layer in enumerate(self.layers):

                # For now, this only operates on kernels in convolutional layers
                if layer.is_conv:

                    for node_index, node in enumerate(layer.nodes):

                        # Check how many links we would have to add or erase
                        for dim in range(len(layer.kernel_size)):

                            if node.size(dim + 1) > 1:

                                #print("Number of links affected by shrinking dimension", dim, "of kernel", node_index, "in layer", layer_index)

                                link_count = 2 * layer.get_input_nodes() * math.pow(node.size(dim + 1), len(layer.kernel_size) - 1)

                                #print("\t>>> Total:", link_count)

                                # Check how many links we would have to erase
                                wheel.add((layer_index, node_index, dim), 1 / link_count)

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
        delta = {}
        for dim in _delta.keys():
            delta[dim] = -abs(_delta[dim])
            
        print(">>> Shrinking dimension(s)", *delta.keys(), "of kernel", _node_index, "in layer", _layer_index, "by", abs(*delta.values()))

        # Shrink the kernel
        success = self.layers[_layer_index].resize_kernel(_node_index, delta)
    
        return (success, _layer_index, _node_index, delta)

    def forward(self,
                _tensor): # Input tensor

        for layer in self.layers:
            _tensor = layer.forward(_tensor)

        return tnf.log_softmax(_tensor, dim = 1)

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
        assert len(_p1.layers) > 0, "!!! Error: Parent %r is empty." % _p1.ID
        assert len(_p2.layers) > 0, "!!! Error: Parent %r is empty." % _p2.ID

        # First, create reference chromosomes.
        # Roulette wheel for selecting chromosomes
        # (layers) and genes (nodes) from the two
        # parents based on their fitness.
        wheel = RouletteWheel()
        wheel.add(_p1, _p1.fitness.relative.current_value)
        wheel.add(_p2, _p2.fitness.relative.current_value)

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
            if (_p1.layers[_p1.cur_layer].type == 
                _p2.layers[_p2.cur_layer].type):

                # Store the parents' layers into the reference genotypes.
                dna1.append(_p1.layers[_p1.cur_layer])
                dna2.append(_p2.layers[_p2.cur_layer])
            
                _p1.cur_layer += 1
                _p2.cur_layer += 1

            else:

                # Determine the parent with the longer DNA
                larger_parent = _p1 if Layer.TypeRanks[_p1.layers[_p1.cur_layer].type] < Layer.TypeRanks[_p2.layers[_p2.cur_layer].type] else _p2
                smaller_parent = _p1 if larger_parent.ID == _p2.ID else _p2

                # Spin the wheel and lock it into a random position
                wheel.lock()

                while (larger_parent.layers[larger_parent.cur_layer].type < smaller_parent.layers[smaller_parent.cur_layer].type):

                    # Select a parent at random.
                    # If we select the parent with the longer DNA,
                    # store all subsequent layers of the same type from that parent.
                    if wheel.spin().ID == larger_parent.ID:
                        dna1.append(larger_parent.layers[larger_parent.cur_layer])
                        dna2.append(larger_parent.layers[larger_parent.cur_layer])

                        #print(">>> _p1.cur_layer type", _p1.layers[_p1.cur_layer].type)
                        #print(">>> _p2.cur_layer type", _p2.layers[_p2.cur_layer].type)

                    larger_parent.cur_layer += 1

                # Allow the wheel to spin again
                wheel.unlock()

                # Reset the pointer to the larger parent
                larger_parent = None

        #=================
        # Crossover
        #=================

        # Perform crossover for each pair of chromosomes (layers).
        for layer_index in range(len(dna1)):

            wheel.replace([dna1[layer_index], dna2[layer_index]])

            # An empty list of nodes.
            # This holds copies of the actual weights
            nodes = type(dna1[layer_index].nodes)()
            bias_vals = None if dna1[layer_index].bias is None else []

            # Node counter
            node_idx = 0

            # Pick kernels from the two reference chromosomes.
            while (node_idx < wheel.elements[0].get_output_nodes() and
                   node_idx < wheel.elements[1].get_output_nodes()):

                rnd_layer = wheel.spin()
                # Pick a node from either layer at random
                nodes.append(rnd_layer.nodes[node_idx])
                if bias_vals is not None:
                    bias_vals.append(rnd_layer.bias.data[node_idx])

                    node_idx += 1

                if (node_idx == wheel.elements[0].get_output_nodes() or
                    node_idx == wheel.elements[1].get_output_nodes()):

                    # Optionally store any extra nodes.
                    rnd_layer = wheel.spin()
                    wheel.replace([rnd_layer, rnd_layer])

            # Restore the elements in the roulette wheel
            wheel.replace([dna1[layer_index], dna2[layer_index]])

            # Compute the shape of the new layer
            shape = [0] * len(list(nodes[0].size()))
            shape[0] = len(nodes)
            
            bias_nodes = None
            if bias_vals is not None:
                bias_nodes = tn.Parameter(torch.zeros(len(bias_vals)))
                for idx, val in enumerate(bias_vals):
                    bias_nodes.data[idx] = val

#            print(">>> Layer", layer_index, ":", shape)

            self.add_layer(_shape = shape,
                           _bias = bias_nodes is not None,
                           _layer_index = layer_index,
                           _activation = wheel.spin().activation,
                           _nodes = nodes,
                           _bias_nodes = bias_nodes)

    def mutate(self,
               _structure = True,
               _parameters = True):

        from . import random as Rand 
        from .species import Species
        
        # Statistics about the structure of this network
        stats = Net.get_structure_stats(self.ID)

#        print("\n>>> Network statistics:")
#        for stat in stats.values():
#            stat.print()

        # Complexity can be increased or decreased
        # based on the current complexity of the
        # network relative to the average complexity
        # of the whole population.
        complexify = Rand.chance(1.0 - self.get_parametric_complexity())

        # The complexity can be increased or decreased
        # with probability proportional to the number 
        # of parameters that the mutation will affect.
        wheel = RouletteWheel(WeightType.Inverse)

        if (Species.Max.Count > 0 and
            len(Species.env) == Species.Max.Count):
            _structure = False

        if _structure:
            # Adding or erasing a layer involves severing existing links and adding new ones.
            # For this computation, we assume that the new layer will contain
            # the mean number of output nodes.
            wheel.add('layer', (stats['nodes'].mean + stats['nodes'].get_sd()) * stats['links'].mean)

            if (len(self.layers) > 2 or 
                (len(self.layers) == 2 and
                 self.layers[0].is_conv)):
                # Adding or erasing a node involves adding or erasing new links.
                wheel.add('node', stats['links'].mean)

        if _parameters:
            # Growing or shrinking a kernel involves adding or removing a shell of links
            # around the kernel in one of the available dimensions.
            # This is multiplied by the average number of input nodes.
            if stats['kernel_dims'] > 0:
                wheel.add('kernel', 2 * stats['nodes'].mean * math.pow(stats['kernel_sizes'].mean, stats['kernel_dims'] - 1))

        if wheel.is_empty():
            return False

        element_type = wheel.spin()

        # Perform a random mutation
        
        success = False

        for elem_index in range(len(wheel.elements)):
            print(wheel.elements[elem_index], "|", wheel.weights[WeightType.Raw][elem_index], "|", wheel.weights[WeightType.Inverse][elem_index])            
            
        print("Adding" if complexify else "Erasing", element_type)

        return

        if element_type == 'layer':
            success = self.add_layer(_stats = stats) if complexify else self.erase_layer()

        elif element_type == 'node':
            success = self.add_nodes(_stats = stats) if complexify else self.erase_nodes()

        elif element_type == 'kernel':
            success = self.grow_kernel(_stats = stats) if complexify else self.shrink_kernel()

        if success:
            
            # Check if the mutation was structural
            if (element_type == 'layer' or
                element_type == 'node'):
            
                # Check if speciation is enabled
                if (self.species_id != 0 and
                    self.id != 0):
                
                    # Create a new species
                    new_species = species(_genome = self.get_genome())

                    # Add the network to the new species
                    new_species.nets.add(self.id)

                    # Remove the network from the current species
                    species.env[self.species_id].nets.remove(self.id)

                    # Store the species ID in this network
                    self.species_id = new_species.id

        return success
