import sys

import torch
import torch.nn as tn
import torch.nn.functional as tnf

import cortex.random as Rand
import cortex.functions as Func

class Layer(tn.Module):

    Ops = {
        1: tnf.linear,
        2: tnf.conv1d,
        3: tnf.conv2d,
        4: tnf.conv3d
        }

    Activations = {
        'linear': tnf.leaky_relu,
        'conv1d': tnf.leaky_relu,
        'conv2d': tnf.leaky_relu,
        'conv3d': tnf.leaky_relu,
        'output': tnf.log_softmax
    }

    Roles = {
        'conv3d': 0,
        'conv2d': 1,
        'conv1d': 2,
        'linear': 3,
        'output': 4
        }

    Bias = True
    InitFunction = tn.init.uniform_
    InitArgs = {'a': -0.01, 'b': 0.05}

    ### Layer definition class
    class Def:

        def __init__(self,
                     _shape, # [nodes, depth, height, width]
                     _bias = None,
                     _activation = None,
                     _role = None):

            if isinstance(_shape, int):
                self.shape = [_shape]

            elif isinstance(_shape, list):

                if (len(_shape) == 0 or
                    _shape[0] < 0):
                    self.shape = [1]

                else:
                    self.shape = list(_shape)

            else:
                self.shape = [1]

            self.op = Layer.Ops[len(self.shape)]

            self.bias = Layer.Bias if _bias is None else _bias
            self.role = self.op.__name__ if _role is None else _role
            self.activation = None
            if _activation is not None:
                self.activation =_activation
            elif (self.role is not None and
                  self.role in Layer.Activations):
                self.activation = Layer.Activations[self.role]

            self.is_conv = len(self.shape) > 1
            self.empty = (self.shape[0] == 0)

        def matches(self,
                    _other):

            if isinstance(_other, Layer.Def):
#                print('Shape: {} | {}'.format(self.shape, _other.shape))
#                print('Bias: {} | {}'.format(self.bias, _other.bias))
#                print('Role: {} | {}'.format(self.role, _other.role))
#                print('activation: {} | {}'.format(self.activation, _other.activation))

                return (self.shape == _other.shape and # Node count
                        self.bias == _other.bias and
                        self.role == _other.role and
                        ((self.activation is None and
                          _other.activation is None) or
                          self.activation == _other.activation))
            return False

        def print(self,
                  _file = sys.stdout):

            print("\n\tShape:", self.shape,
                  "\n\tBias:", self.bias,
                  "\n\tOp:", self.op,
                  "\n\tRole:", self.role,
                  "\n\tActivation:", self.activation.__name__,
                  "\n\tConvolutional:", self.is_conv,
                  "\n\tEmpty:", self.empty,
                  file = _file)

        ### /Def

    @staticmethod
    def stretch(_tensor):
        return _tensor.view(-1, Func.prod(list(_tensor.size())[1:]))

    @staticmethod
    def get_centre(_kernel):
        return [dim // 2 for dim in _kernel.size()[1:]]

    @staticmethod
    def compute_output_shape(_output_nodes,
                             _input_shape,
                             _kernel_size = [],
                             _stride = None,
                             _padding = None,
                             _dilation = None):

        """
        Compute the output shape of a layer based on the input shape and the layer's attributes.
        """

        if _padding is None:
            _padding = [dim // 2 for dim in _kernel_size]

        if _dilation is None:
            _dilation = [1] * len(_kernel_size)

        if _stride is None:
            _stride = [1] * len(_kernel_size)

        # The first element is the number of nodes (channels)
        output_shape = [_output_nodes]

        for dim in range(len(_kernel_size)):
            output_shape.append(((_input_shape[dim + 1] + 2 * _padding[dim] - _dilation[dim] * (_kernel_size[dim] - 1) - 1) // _stride[dim] + 1) )

        return output_shape

    @staticmethod
    def init(_tensor):
        if Layer.InitFunction is not None:
            Layer.InitFunction(_tensor, **Layer.InitArgs)

    def __init__(self,
                 _layer_def,
                 _input_shape,
                 _layer_index = None):

        assert 1 <= len(_layer_def.shape) <= 4, "Invalid layer shape %r" % _layer_def.shape
        assert 1 <= len(_input_shape) <= 4, "Invalid input shape %r" % _input_shape
        assert len(_layer_def.shape) == 1 or len(_layer_def.shape) == len(_input_shape), "Invalid input shape %r for layer shape %r" % (_input_shape, _layer_def.shape)
        assert _layer_def.shape[0] >= 0, "Invalid number of nodes (%r) provided for layer %r" % (_layer_def.shape[0], _layer_index)
        assert _input_shape[0] > 0, "Invalid number of input nodes (%r) provided for layer %r" % (_input_shape[0], _layer_index)

        # Initialise the base class
        super(Layer, self).__init__()

        # Store the layer index
        self.index = _layer_index

        # Store the input shape as an attribute.
        # This can change as the containing network evovles.
        self.input_shape = _input_shape

        # Operation to perform on the input tensor
        self.op = _layer_def.op

        # Layer role (string)
        self.role = _layer_def.role

        # Activation function
        self.activation = _layer_def.activation

        # Convenience indicator for whether this layer is convolutional
        self.is_conv = _layer_def.is_conv

        # Common attributes
        self.kernel_size = list(_layer_def.shape[1:]) if self.is_conv else []
        assert (len(self.kernel_size) == 0 or len(self.kernel_size) == len(self.input_shape[1:])), "Invalid kernel size %r for input shape %r" % (self.kernel_size, _input_shape)

        self.stride = [1] * len(self.kernel_size) if self.is_conv else []
        self.padding = [dim // 2 for dim in self.kernel_size] if self.is_conv else []
        self.dilation = [1] * len(self.kernel_size) if self.is_conv else []

        # Used for overlaying kernels onto the weight tensor
        self.weight_slices = []

        # Learnable parameters
        # Convolutional layers store their parameters
        # in a list of variables which are overlaid onto
        # the layer's weights, whereas linear layers store
        # their parameters directly as weights.
        self.weight = torch.Tensor() if self.is_conv else tn.Parameter()
        self.nodes = tn.ParameterList() if self.is_conv else []
        self.bias = None if not _layer_def.bias else tn.Parameter()

        if not _layer_def.empty:
            # Generate nodes
            self.add_nodes(_layer_def.shape[0],
                           self.input_shape[1:],
                           _layer_def.shape[1:])

    def matches(self,
                _other):

        tolerance = 1e-8

        self.update_nodes()
        _other.update_nodes()

        if len(self.nodes) != len(_other.nodes):
            print("(Layer", self.index, ")\t>>> Different number of nodes")
            print("\t>>> Layer 1:\n", self.nodes)
            print("\t>>> Layer 2:\n", _other.nodes)
            return False

        if (self.role != _other.role):
            print("(Layer", self.index, ")\t>>> Different layer roles")
            print("\t>>> Layer 1:\n", self.role)
            print("\t>>> Layer 2:\n", _other.role)
            return False

        if self.activation != _other.activation:
            print("(Layer", self.index, ")\t>>> Different activations")
            print("\t>>> Layer 1:\n", self.activation)
            print("\t>>> Layer 2:\n", _other.activation)
            return False

        if self.input_shape != _other.input_shape:
            print("(Layer", self.index, ")\t>>> Different input shapes")
            print("\t>>> Layer 1:\n", self.input_shape)
            print("\t>>> Layer 2:\n", _other.input_shape)
            return False

        if self.get_input_nodes() != _other.get_input_nodes():
            print("(Layer", self.index, ")\t>>> Different input node count")
            print("\t>>> Layer 1:\n", self.get_input_nodes())
            print("\t>>> Layer 2:\n", _other.get_input_nodes())
            return False

        if self.get_output_shape() != _other.get_output_shape():
            print("(Layer", self.index, ")\t>>> Different output shapes")
            print("\t>>> Layer 1:\n", self.output_shape())
            print("\t>>> Layer 2:\n", _other.output_shape())
            return False

        if self.get_output_nodes() != _other.get_output_nodes():
            print("(Layer", self.index, ")\t>>> Different output node count")
            print("\t>>> Layer 1:\n", self.get_output_nodes())
            print("\t>>> Layer 2:\n", _other.get_output_nodes())
            return False

        if self.get_multiplier() != _other.get_multiplier():
            print("(Layer", self.index, ")\t>>> Different multiplier")
            print("\t>>> Layer 1:\n", self.get_multiplier())
            print("\t>>> Layer 2:\n", _other.get_multiplier())
            return False

        if self.kernel_size != _other.kernel_size:
            print("(Layer", self.index, ")\t>>> Different kernel size")
            print("\t>>> Layer 1:\n", self.kernel_size)
            print("\t>>> Layer 2:\n", _other.kernel_size)
            return False

        if self.padding != _other.padding:
            print("(Layer", self.index, ")\t>>> Different padding")
            print("\t>>> Layer 1:\n", self.padding)
            print("\t>>> Layer 2:\n", _other.padding)
            return False

        if self.stride != _other.stride:
            print("(Layer", self.index, ")\t>>> Different stride")
            print("\t>>> Layer 1:\n", self.stride)
            print("\t>>> Layer 2:\n", _other.stride)
            return False

        if self.dilation != _other.dilation:
            print("(Layer", self.index, ")\t>>> Different dilation")
            print("\t>>> Layer 1:\n", self.dilation)
            print("\t>>> Layer 2:\n", _other.dilation)
            return False

        if ((self.bias is None and
             _other.bias is not None)
            or
            (self.bias is not None and
             _other.bias is None)
            or
            (self.bias is not None and
             _other.bias is not None and
             not torch.allclose(self.bias, _other.bias, tolerance, tolerance))):
                print("(Layer", self.index, ")\t>>> Different bias")
                print("\t>>> Layer 1:\n", self.bias)
                print("\t>>> Layer 2:\n", _other.bias)
                return False

        for node_index in range(len(self.nodes)):
            if not torch.allclose(self.nodes[node_index], _other.nodes[node_index], tolerance, tolerance):
                print("(Layer", self.index, ")\t>>> Different node in position", node_index)
                print("\t>>> Layer 1:\n", self.nodes[node_index])
                print("\t>>> Layer 2:\n", _other.nodes[node_index])
                return False

        return True

    def print(self,
              _file = sys.stdout):

        print("\n==================[ Layer", self.index, "]==================",
              "\n>>> Role:", self.role,
              "\n>>> Activation:", self.activation.__class__.__name__,
              "\n>>> Input shape:", self.input_shape, ", input nodes:", self.get_input_nodes(), ", multiplier:", self.get_multiplier(),
              "\n>>> Output shape:", self.get_output_shape(), ", output nodes:", self.get_output_nodes(),
              "\n>>> Size of weight tensor:\n", self.weight.size(),
              "\n>>> Size of bias tensor:\n", self.bias.size(),
              "\n>>> Attributes:",
              "\n\tKernel size:", self.kernel_size,
              "\n\tStride:", self.stride,
              "\n\tPadding:", self.padding,
              "\n\tDilation:", self.dilation,
              "\n>>> Learnable parameters:",
              file = _file)

        for idx, param in enumerate(self.parameters()):
            print("\n>>> Parameter", idx, ":\n", param, file = _file)

    def get_multiplier(self,
                       _input_shape = None):
        if _input_shape is None:
            _input_shape = self.input_shape

        #print("product over", _input_shape[1:len(_input_shape) - len(self.kernel_size)])
        return Func.prod(_input_shape[1:len(_input_shape) - len(self.kernel_size)])

    def get_input_nodes(self,
                        _input_shape = None):
        if _input_shape is None:
            _input_shape = self.input_shape
        return _input_shape[0]

    def get_output_nodes(self):
        return len(self.nodes)

    def get_output_shape(self,
                         _input_shape = None):

        if _input_shape is None:
            _input_shape = self.input_shape
        return Layer.compute_output_shape(self.get_output_nodes(),
                                          _input_shape,
                                          self.kernel_size,
                                          self.stride,
                                          self.padding,
                                          self.dilation)

    def get_parameter_count(self,
                       _node_idx = None):

        nodes = [node for node in range(len(self.nodes))] if _node_idx is None else [_node_idx]

        parameters = 0
        for node_idx in nodes:
            parameters += self.nodes[node_idx].numel()

        return parameters

    def get_mean_parameter_count(self):
        return self.get_parameter_count() / len(self.nodes)

    def get_random_kernel_sizes(self,
                                _count,
                                _max_radius = [],
                                _kernel_size = []): # Pre-determined kernel size. Dimensions with value 0 are populated with random values.

        assert len(_kernel_size) == 0 or len(_kernel_size) == len(_max_radius), "Invalid kernel size %r" % _kernel_size

        wheel = Rand.RouletteWheel()

        #print("Kernel size:", _kernel_size)

        if self.is_conv:

            if len(_max_radius) == 0:
                _max_radius = [dim for dim in self.input_shape[1:]]

            # Possible extents for the kernel size.
            extents = []
            for dim, radius in enumerate(_max_radius):

                if (len(_kernel_size) > 0 and
                    _kernel_size[dim] > 0):
                    ext = [_kernel_size[dim]]

                else:
                    if radius <= 1:
                        ext = [1]

                    else:
                        ext = [size for size in range(1, radius // 2 + 1) if size % 2 == 1]

                if len(extents) == 0:
                    extents = [[e] for e in ext]

                else:
                    new_extents = []

                    for old_ext in extents:
                        for new_ext in ext:
                            new_extents.append([*old_ext, new_ext])

                    extents = new_extents

            #print("Extents:", extents)

            for e in extents:
                wheel.add(e, Func.exp_prod(e))

        else:
            # Dummy zero-dimensional kernel
            wheel.add([], 1)

#        for idx in range(len(wheel.elements)):
#            print(wheel.elements[idx], "\t", wheel.weights[Rand.WeightType.Raw][idx], "\t", wheel.weights[Rand.WeightType.Inverse][idx])

        return [wheel.spin() for k in range(_count)]

    def add_nodes(self,
                  _count,
                  _max_radius = [],
                  _kernel_size = []):

        if _count <= 0:
            return False

        # Weights -> nodes
        self.update_nodes()

        # Generate a list of random kernel sizes
        if self.is_conv:
            kernel_sizes = self.get_random_kernel_sizes(_count, _max_radius, _kernel_size)

#        print('Original node count: {}'.format(len(self.nodes)))

        # Append the nodes
        for node in range(_count):

            if self.is_conv:
                self.nodes.append(tn.Parameter(torch.zeros(self.get_input_nodes() * self.get_multiplier(), *kernel_sizes[node])))
                #print("(Conv) New node with size", self.nodes[-1].size(), ", multiplier", self.get_multiplier())

            else:
                self.nodes.append(tn.Parameter(torch.zeros(self.get_input_nodes() * self.get_multiplier())))
                #print("(FC) New node with size", self.nodes[-1].size(), ", multiplier", self.get_multiplier())

            Layer.init(self.nodes[-1])

        # Add bias nodes
        if self.bias is not None:
            bias = torch.zeros(_count)
            Layer.init(bias)
#                    print("resize() extra bias:", bias)
            self.bias = tn.Parameter(torch.cat((self.bias.clone().detach(), bias)))

        # Nodes -> weights
        self.update_weights()

#        print('New node count: {}'.format(len(self.nodes)))

        return True

    def erase_nodes(self,
                    _node_indices = []):

        if (len(_node_indices) == 0 or
            len(_node_indices) >= len(self.nodes)):
            return False

        # Weights -> nodes
        self.update_nodes()

        #print("Nodes to erase:", *_node_indices)

#        print('Original node count: {}'.format(len(self.nodes)))

        # Erase the selected nodes
        nodes = tn.ParameterList()
        bias = []

        for index, node in enumerate(self.nodes):
            if index not in _node_indices:
                nodes.append(node)
                if bias is not None:
                    bias.append(self.bias[index].item())

        # Update the bias parameters
        if len(bias) > 0:
            self.bias = tn.Parameter(torch.Tensor(bias))

        self.nodes = nodes

#        print('New node count: {}'.format(len(self.nodes)))

        # Nodes -> weights
        self.update_weights()

        return True

    def resize_kernel(self,
                      _node_index,
                      _delta):

        # Sanity checks
        if (not self.is_conv or
            not (_node_index >= 0 and _node_index < len(self.nodes)) or
            len(_delta) != 1):
            return False

        delta = [0] * len(self.kernel_size)
        for dim, val in _delta.items():
            delta[dim] = val

        old_size = list(self.nodes[_node_index].size()[1:])
        new_size = [old_size[dim] + 2 * delta[dim] for dim in range(len(old_size))]

    #    print(_layer.kernels)
    #    print(_layer.weight[_node_index])

        # Apply padding to the kernel and initialise the
        # new weights if the padding is positive.
        padding = [0] * 2 * len(new_size)
        init = False
        slices1 = [slice(None, None)] # Front, top or left
        slices2 = [slice(None, None)] # Back, bottom or right

        for dim in range(len(delta)):
            padding[2 * (len(new_size) - dim - 1) : 2 * (len(new_size) - dim)] = [delta[dim], delta[dim]]
            if delta[dim] > 0:
                init = True
                slices1.append(slice(0, delta[dim]))
                slices2.append(slice(-delta[dim], None))
            else:
                slices1.append(slice(0, None))
                slices2.append(slice(0, None))

        #print("\n============================================")
        #print(">>> Resizing dimension", *_delta.keys(), "of kernel", _node_index, "by", *_delta.values())
        #print(">>> \tpadding:", padding)
        #print(">>> \told_size:", old_size)
        #print(">>> \tdelta:", delta)
        #print(">>> \tnew_size:", new_size)
        #print("============================================\n")

        #print(">>> Before resize:")
        #print(">>> \tkernel:", self.nodes[_node_index])

        self.nodes[_node_index] = tn.Parameter(tnf.pad(self.nodes[_node_index], padding))

        if init:
            # Initialise the new weights
            Layer.init(self.nodes[_node_index][slices1])
            Layer.init(self.nodes[_node_index][slices2])

        #print(">>> After resize:")
        #print(">>> \tkernel:", self.nodes[_node_index])

        # Create weights from nodes
        self.update_weights()

        return True

    def adjust_input_size(self,
                          _input_shape = None,
                          _node_indices = set(),
                          _pretend = False):

        if not _pretend:
            if _input_shape is not None:
                # Store the new input shape
                self.input_shape = list(_input_shape)

            # Update the layer kernel
            self.update_kernel()

            # Update the nodes so we can manipulate them
            self.update_nodes()

        if _input_shape is None:
            # Adopt the existing input shape
            _input_shape = list(self.input_shape)

        multiplier = self.get_multiplier(_input_shape)
        actual_input_nodes = self.get_input_nodes(_input_shape)

        #print("Layer", self.index, "output shape:", self.get_output_shape())
        #print("adjust_input_size() input shape:", _input_shape)
        #print("adjust_input_size() nodes:", len(self.nodes))
        #print("adjust_input_size() actual_input_nodes:", actual_input_nodes)
        #print("adjust_input_size() multiplier:", multiplier)
        #print(">>> Actual input nodes:", actual_input_nodes)

        if _pretend:
            link_diff = 0

        for output_node in range(self.get_output_nodes()):

            # Check if there is a discrepancy in the number of input nodes
            input_nodes = int(self.nodes[output_node].size(0)) / multiplier
            input_node_diff = input_nodes - actual_input_nodes

            #print(">>> node", output_node, " input_nodes:", input_nodes)
            #print(">>> difference:", input_node_diff)

            if _pretend:

                #print("node", output_node, "size:", self.nodes[output_node].size())
                #print("link difference:", abs(input_node_diff) * self.nodes[output_node][0].numel() * multiplier)

                link_diff += abs(input_node_diff) * Func.prod(list(self.nodes[output_node].size())[1:]) * multiplier

            else:

                if input_node_diff > 0:

                    # There are more input nodes in this layer than
                    # there are output ones in the preceding one.
                    # Shrink the receptive fields of all nodes.

                    if len(_node_indices) == 0:

                        #print("Clipping node", output_node, "to have input size of", actual_input_nodes * multiplier)
                        self.nodes[output_node] = tn.Parameter(self.nodes[output_node][0:actual_input_nodes * multiplier])
                        #print("Node", output_node, "size:", self.nodes[output_node].size())

                    else:

                        #print("Expanding node", output_node, "to have input size of", actual_input_nodes * multiplier)

                        slices = []

                        begin = 0
                        for node_index in _node_indices:

                            if node_index == begin:
                                begin += 1

                            else:
                                slices.append(slice(begin * multiplier, node_index * multiplier))
                                begin = node_index + 1

                        if begin < input_nodes:
                            slices.append(slice(begin * multiplier, None))

                        #print("Slices:", slices)

                        new_node = self.nodes[output_node][slices[0]].clone().detach()

                        for slice_index in range(1, len(slices)):
                            new_node = torch.cat((new_node, self.nodes[output_node][slices[slice_index]].clone().detach()))

                        self.nodes[output_node] = tn.Parameter(new_node)

                elif input_node_diff < 0:

                    # Take the absolute value of the difference
                    input_node_diff = abs(input_node_diff)

                    # There are more output nodes in the preceding layer
                    # than there are input ones in this one.
                    # Expand the receptive fields of all nodes.
                    #print("Expanding node", output_node, "to have input size of", actual_input_nodes * multiplier)

                    padding = torch.zeros(int(actual_input_nodes * multiplier - self.nodes[output_node].size(0)), *list(self.nodes[output_node].size())[1:])
                    Layer.init(padding)

                    self.nodes[output_node] = tn.Parameter(torch.cat((self.nodes[output_node].clone().detach(), padding)))

        if _pretend:
            return link_diff

        else:
            # Update weight data from nodes
            self.update_weights()

    def update_kernel(self):

        """
        Update the layer kernel size (only for convolutional layers)
        """

        if self.is_conv:
            sizes = [0] * len(self.kernel_size)
            for node in self.nodes:
    #            print(">>> node:", node)
                for dim in range(len(self.kernel_size)):
    #                print(">>> dim, size:", dim, size)
                    if int(node.size(dim + 1)) > sizes[dim]:
                        sizes[dim] = int(node.size(dim + 1))

            self.kernel_size = sizes
#            print("New kernel size:", self.kernel_size)

            # Compute the padding
            self.padding = [size // 2 for size in self.kernel_size]

    def update_nodes(self):
        """
        Create nodes from weights.
        For now, this does something only for non-convolutional layers.
        """

        # Update the node data from the weights for non-convolutional layers
        if (not self.is_conv and
            int(self.weight.size(0)) > 0):
            for node_index in range(len(self.nodes)):
                self.nodes[node_index] = tn.Parameter(self.weight[node_index].clone().detach().requires_grad_(self.is_conv))

    def update_slices(self):

        if not self.is_conv:
            return

        self.weight_slices = []

        for output_node in range(len(self.nodes)):
            self.weight_slices.append([slice(0, None)])

            for dim, size in enumerate(list(self.nodes[output_node].size())[1:]):
                offset = (self.kernel_size[dim] - size) // 2
                self.weight_slices[-1].append(slice(offset, self.kernel_size[dim] - offset))

#        print("Weight slices:", self.weight_slices)
#        print("Weights:\n", self.weight.size())
#        print("Nodes:\n", self.nodes)

    def update_weights(self):
        """
        Create weights from nodes.
        """

        # Update the weight tensor
        if self.is_conv:

            #print("Updating weights")
            # Update the kernel size
            self.update_kernel()

            self.update_slices()

            self.weight = torch.zeros(len(self.nodes),       # Output node count
                                      self.input_shape[0],   # Input node count
                                      *self.kernel_size)     # Unpacked kernel dimensions

        else:
            # We don't want the weights for non-convolutional layers
            # to be generated on the fly like they are in convolutional ones.
            # Instead, the weights are updated directly, which means
            # that manipulating non-convolutional nodes should be preceded
            # by a call to update_nodes() in order to update the
            # nodes from the current weights.
#            self.weight = tn.Parameter(torch.stack(list(self.nodes)).clone().detach().requires_grad_(True))
            tensor_list = []
            for node in self.nodes:
                tensor_list.append(node.clone().detach().requires_grad_(False))

            self.weight = tn.Parameter(torch.stack(tensor_list))

        # Update the requires_grad attribute of all nodes
        for node_id in range(len(self.nodes)):
            if not self.is_conv:
                self.nodes[node_id].requires_grad = False
                self.nodes[node_id].detach_()

    def extract_patch(self,
                      _node_idx,
                      _size = None):

        if not self.is_conv:
            return torch.Tensor()

        slices = [slice(0, self.weight.size(1))]

        if _size is None:
            _size = list(self.nodes[_node_idx].size()[1:])

        for dim in range(len(_size)):
            kernel_size = list(self.weight.size()[2:])
            diff = kernel_size[dim] - _size[dim]

            # Patch is smaller in this dimension, narrow down
            if diff > 0:
                slices.append(slice(diff // 2, kernel_size[dim] - diff // 2))
            else:
                slices.append(slice(0, kernel_size[dim]))

        return self.nodes[_node_idx][slices].clone().detach().requires_grad_(False)

    def overlay_kernels(self):

        self.weight = self.weight.detach()

        for output_node in range(len(self.nodes)):
            self.weight[output_node][self.weight_slices[output_node]] = self.nodes[output_node]

    def forward(self,
                _tensor):

        #print("layer", self.index, "forward()")
        #print("weight tensor size:", self.weight.size())
        #print("output shape:", self.get_output_shape())

        if self.is_conv:
            self.overlay_kernels()
#            self.weight[self.weight_slices] = self.nodes
            _tensor = self.op(_tensor, self.weight, self.bias, self.stride, self.padding, self.dilation)

        else:
            if len(list(_tensor.size())) > 2:
                _tensor = Layer.stretch(_tensor)
            _tensor = self.op(_tensor, self.weight, self.bias)

        if self.activation is not None:
            _tensor = self.activation(_tensor)

        return _tensor
