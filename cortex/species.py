# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:54:42 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

class Species:

    class Init:
        Count = 5
    
    class Max:
        Count = 15
        
    env = {}
    ID = 0

    @staticmethod
    def exists(_genome):
        for genome in Species.env.values():
            if genome == _genome:
                return True
            
        return False

    def __init__(self,
                 _genome = None,
                 _other = None,
                 _isolated = False):

        from .network import Net
        from .fitness import Fitness

        self.ID = 0
        
        # Increment the global species ID
        if not _isolated:
            Species.ID += 1
            self.ID = Species.ID
            Species.env[self.ID] = self
        
        # Species champion
        self.champ = None
        
        # Overall species fitness computed from the fitness 
        # of networks belonging to this species
        self.fitness = Fitness()
        
        # Set of networks belonging to this species
        self.nets = set()
        
        # Species genome (list of layer definitions)
        self.genome = []
        
        if _other is None:
#            print("Creating genome from initial layer definitions")
            self.genome = list(Net.Init.Layers) if _genome is None else list(_genome)
        
        else:
#            print("Copying genome from species", _other.ID)
            self.genome = list(_other.genome)

        print(">>> Species", self.ID, "created")

    def __eq__(self,
               _other):
        """
        Equality testing for species
        """
        other_genome = None
        
        from .layer import Layer
        
        if isinstance(_other, list):
#            print("Comparing species", self.ID, "with a raw genome")
#            print("Species", self.ID, "genome:", len(self.genome), "Genome:", len(other_genome))
            other_genome = _other
            
        elif isinstance(_other, Species):
#            print("Comparing species", self.ID, "with species", _other.ID)
#            print("Species", self.ID, "genome:", len(self.genome), "Species", _other.ID, "genome:", len(_other.genome))
            other_genome = _other.genome
            
        if other_genome is None:
            return False
            
        if len(self.genome) != len(other_genome):
            return False

        for layer_index in range(len(self.genome)):
#                print("Comparing layer ", l)
            
            if not (isinstance(self.genome[layer_index], Layer.Def) and
                    isinstance(other_genome[layer_index], Layer.Def) and
                    self.genome[layer_index] == other_genome[layer_index]):
                return False
            
        return True

    def __ne__(self, _other):
        return not (self == _other)

    def calibrate(self):

        from .network import Net
        from .fitness import Fitness
        from . import statistics as Stat

        if len(self.nets) == 0:
            return (0.0, None)

        net_stats = Stat.SMAStat()
        max_fit = -1.0

        # Compute the absolute fitness of the species
        for net_id in self.nets:
            abs_fit = Net.population[net_id].fitness.absolute.value

            if abs_fit > max_fit:
                max_fit = abs_fit

            net_stats.update(abs_fit)

        # Sort the networks in order of decreasing fitness
        self.nets = sorted(self.nets, key = lambda net_id: Net.population[net_id].fitness.absolute.value, reverse = True)

        # Compute the relative fitness of the networks
        # belonging to this species
        for net_id in self.nets:
            # Compute the relative fitness
           Net.population[net_id].fitness.calibrate(net_stats)

           print("Network ", net_id, " fitness:",
                 "\t\tAbsolute: ", Net.population[net_id].fitness.absolute.value,
                 "(mean", Net.population[net_id].fitness.absolute.mean,
                 ", sd", Net.population[net_id].fitness.absolute.sd() + ")"

                 "\t\tRelative: ", Net.population[net_id].fitness.relative.value,
                 "(mean", Net.population[net_id].fitness.relative.mean,
                 ", sd", Net.population[net_id].fitness.relative.sd() + ")")

        self.fitness.absolute.update(net_stats.mean)

        # Return the champion for this species and the genome fitness
        return (self.fitness.absolute.value, self.nets[0] if len(self.nets) > 0 else None)

#    def mutate(self):
#        print(">>> Mutating genome", self.ID, "...")
#
#        by_fixed = {}
#        by_type = {}
#        by_nodes = {}
#
#        mut = Mutation()
#
#        for l in range(0, len(self.genome)):
#
#            if not self.genome[l].is_fixed in by_fixed:
#                by_fixed[self.genome[l].is_fixed] = []
#            by_fixed[self.genome[l].is_fixed].append(l)
#
#            if not self.genome[l].type in by_type:
#                by_type[self.genome[l].type] = []
#            by_type[self.genome[l].type].append(l)
#
#            if not self.genome[l].nodes in by_nodes:
#                by_nodes[self.genome[l].nodes] = []
#            by_nodes[self.genome[l].nodes].append(l)
#
#        if not False in by_fixed:
#            print(">>> No evolvable layers.")
#            # No evolvable layers. Add one.
#            mut.element = ctx.ElemType.Layer
#            mut.action = mConf.Action.Inc
#
#        else:
#            # Compute the probability of adding
#            # a node as the deviation from the
#            # average number of nodes across
#            # the environment.
#            add_node_chance = 1.0 - self.complexity()
#
#            print(">>> Chance of adding a node:", add_node_chance)
#
#            # There are evolvable layers.
#            # The default element to add in this case is a node.
#            # However, there is an additional check to see if
#            # we should add a layer (with a node) instead.
#            mut.element = mConf.ElemType.Node
#
#            # Determine whether we are adding or erasing a node.
#            # The probability of adding a node is inversely
#            # proportional to the genome's complexity.
#            mut.action = mConf.Action.Inc if mRand.chance(add_node_chance) else mConf.Action.Dec
#
#            # Check whether we should add    a new layer.
#            # This depends on the complexity of the
#            # genome and the total number of layers.
#            add_layer_chance = math.pow(add_node_chance, len(self.genome))
#
#            print(">>> Chance of adding a layer:", add_layer_chance)
#
#            if (mut.action == mConf.Action.Inc and
#                mRand.chance(add_layer_chance)):
#                mut.element = mConf.ElemType.Layer
#
#        # Conditionally count the nodes while
#        # populating the roulette wheel.
#        removable_nodes = {}
#        for l in range(0, len(self.genome)):
#            if not self.genome[l].Fixed:
#                removable_nodes[l] = self.genome[l].nodes
#
#        count_nodes = (not False in by_fixed or
#                       len(removable_nodes) == 0)
#
#        if count_nodes:
#            print(">>> Counting nodes")
#
#        # Special check in case we are trying to remove
#        # a layer from a minimal network.
#        if (count_nodes and
#            mut.element == mConf.ElemType.Layer and
#            mut.action == mConf.Action.Dec):
#
#            mut.action = mConf.Action.Inc
#
#        # Roulette wheel for selecting a layer to mutate
#        wheel = RouletteWheel()
#        for l in range(0, len(self.genome)):
#            if (# In case of adding a layer, we are only
#                # trying to decide its type.
#                # Include all layers in the wheel.
#                (mut.element == mConf.ElemType.Layer and
#                 mut.action == mConf.Action.Inc)
#                or
#                # The layer is evolvable and either has
#                # two or more nodes or we we are not
#                # checking the node count.
#                not self.genome[l].Fixed):
#                    wheel.add(l, self.genome[l].nodes)
#
#        if wheel.is_empty():
#            print("Wheel is empty")
#            return None
#
#        print("\nWheel size:", len(wheel.elements))
#
#        mut.layer_index = wheel.spin(WeightType.Inverse) if mut.action == mConf.Action.Inc else wheel.spin()
#
#        # Choose the layer to mutate.
#
#        print("\nLayer index:", mut.layer_index)
#
#        # Get the layer type from the selected layer.
#        # This is used when adding a layer to determine
#        # what type the new layer should be.
#        mut.layer_type = self.genome[mut.layer_index].Type
#
#        # If we are removing the last node of a layer,
#        # remove the whole layer.
#        if (mut.element == mConf.ElemType.Node and
#            mut.action == mConf.Action.Dec and
#            self.genome[mut.layer_index].nodes == 1):
#            mut.element = mConf.ElemType.Layer
#
#        if (mut.element == mConf.ElemType.Layer and
#            mut.action == mConf.Action.Inc):
#            mut.layer_index = by_type[self.genome[mut.layer_index].Type][-1]
#
#        self.apply(mut)
#
#        return mut
#
#    def apply(self, _mut):
#
#        print("Applying mutation to genome", self.ID)
#
#        if _mut is not None:
#            # Apply the mutation if it is valid
#
#            if _mut.element == mConf.ElemType.Node:
#                if _mut.action == mConf.Action.Inc:
#                    self.genome[_mut.layer_index].nodes += 1
#                else:
#                    self.genome[_mut.layer_index].nodes -= 1
#
#            elif _mut.element == mConf.ElemType.Layer:
#                if _mut.action == mConf.Action.Inc:
#                    self.genome.insert(_mut.layer_index, mConf.LayerDef(_mut.layer_type, 1, False))
#                else:
#                    del self.genome[_mut.layer_index]
#
#        print("Genome", self.ID, "after mutation:")
#        for l in range(0, len(self.genome)):
#            print("Layer", l, ":", self.genome[l].Type.name, ", ", self.genome[l].Fixed, ", ", self.genome[l].nodes)
#
#    def complexity(self):
#
#        # Collect statistics about node count for all genomes
#        node_stat = mStat.SMAStat()
#        for genome in mConf.genomes.values():
#            n_count = 0
#            for layer in genome.genome:
#                n_count += layer.nodes
#            node_stat.update(n_count)
#
#        # Get the number of nodes in this genome
#        n_count = 0
#        for layer in self.genome:
#            n_count += layer.nodes
#
#        # The complexity is computed as the deviation from
#        # the average number of nodes across all genomes.
#        return node_stat.offset(n_count)
