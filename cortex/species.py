import math
import sys
from copy import deepcopy as dcp

import cortex.random as Rand
import cortex.statistics as Stat
from cortex.fitness import Fitness

class Species:

    # Static members
    Enabled = True

    class Init:
        Count = 4

    class Max:
        Count = 16

    ID = 0
    Offspring = 0
    Populations = {}

    @staticmethod
    def find(_genome):
        for species in Species.Populations.values():
            if species.matches(_genome):
                return species.ID

        return 0

    @staticmethod
    def reset():
        Species.ID = 0
        Species.Offspring = 0
        Species.Populations = {}

    def __init__(self,
                 _genome = None,
                 _other = None,
                 _isolated = False):

        import cortex.network as cn

        self.ID = 0

        # Increment the global species ID
        if not _isolated:
            Species.ID += 1
            self.ID = Species.ID
            Species.Populations[self.ID] = self

        # Species champion
        self.champion = None

        # Overall species fitness computed from the fitness
        # of networks belonging to this species
        self.fitness = Fitness()

        # Set of networks belonging to this species
        self.nets = set()

        # Species genome (list of layer definitions)
        self.genome = []

        if _other is None:
#            print("Creating genome from initial layer definitions")
            self.genome = dcp(cn.Net.Init.Layers) if _genome is None else dcp(_genome)

        else:
#            print("Copying genome from species", _other.ID)
            self.genome = dcp(_other.genome)

        print(">>> Species", self.ID, "created")

    def matches(self,
                _other):
        """
        Equality testing for species / genomes
        """

        other_genome = None
        if isinstance(_other, list):
            other_genome = _other
            #print("Comparing species", self.ID, "with a raw genome")

        elif isinstance(_other, Species):
            other_genome = _other.genome
            #print("Comparing species", self.ID, "with species", _other.ID)

        assert other_genome is not None, 'Species comparison operation failed: comparing {} with {}'.format(self, _other)

        if not Species.Enabled:
            # If speciation is disabled, any genome compares equal to any other genome
            return True

        #print("Genome 1 length:", len(self.genome), ", Genome 2 length:", len(other_genome))

        if len(self.genome) != len(other_genome):
            return False

        for layer_index in range(len(self.genome)):
            if not self.genome[layer_index].matches(other_genome[layer_index]):
                return False

        return True

    def print(self,
              _file = sys.stdout):

        import cortex.layer as cl
        import cortex.network as cn

        print("\n\n===============[ Species", self.ID, "]===============",
              "\nAbsolute fitness:", self.fitness.absolute,
              "\nRelative fitness:", self.fitness.relative,
              "\nNetworks:", self.nets,
              "\nchampion:", self.champion,
              "\n====[ Genome ]====",
              file = _file)

        for layer_index, layer in enumerate(self.genome):
            print('Layer {}:'.format(layer_index), file = _file)
            layer.print(_file = _file)

        output_layer = cl.Layer.Def(_shape = cn.Net.Output.Shape,
                                    _role = 'output')

        print('Layer {}:'.format(len(self.genome)), file = _file)
        output_layer.print(_file = _file)

    def calibrate(self,
                  _complexity_fitness_scale):

        if len(self.nets) == 0:
            return

        import cortex.network as cn

        # Reset the champion
        self.champion = None

        net_stats = Stat.SMAStat()
        top_fitness = -math.inf

        # Compute the absolute fitness of the species
        for net_id in self.nets:

            absolute_fitness = cn.Net.Ecosystem[net_id].fitness.absolute

            if (self.champion is None or
                absolute_fitness > top_fitness):
                self.champion = net_id
                top_fitness = absolute_fitness

            net_stats.update(absolute_fitness)

#        print("Networks for species", self.ID, "sorted in order of descending fitness:", sorted_nets)
        print("champion for species", self.ID, ":", self.champion)

        # Compute the relative fitness of the networks
        # belonging to this species
        for net_id in self.nets:
            # Compute the relative fitness
            net = cn.Net.Ecosystem[net_id]
            net.fitness.relative = _complexity_fitness_scale[net_id] * net_stats.get_offset(net.fitness.absolute)
#            net.fitness.relative = net_stats.get_offset(net.fitness.absolute)

            print("Network", net_id, "fitness:",
                  "\t\tAbsolute:", cn.Net.Ecosystem[net_id].fitness.absolute,
                  "\t\tRelative:", cn.Net.Ecosystem[net_id].fitness.relative)

        self.fitness.absolute = net_stats.mean

    def evolve(self):

        import cortex.network as cn

        # Populate the parent wheel.
        # Networks that have been selected for crossover
        # will pick a partner at random by spinning the wheel.
        parent_wheel = Rand.RouletteWheel()
        for net_id in list(self.nets):
            parent_wheel.add(net_id, cn.Net.Ecosystem[net_id].fitness.relative)

        # Iterate over the networks and check if we should perform crossover or mutation
        for net_id in list(self.nets):
            if (Species.Offspring < len(cn.Net.Ecosystem) // 2 and
                Rand.chance(cn.Net.Ecosystem[net_id].fitness.relative)):

                # Fitter networks have a better chance of mating
                p2 = cn.Net.Ecosystem[parent_wheel.spin()]
                if p2.ID != net_id:
                    # Crossover
                    offspring = cn.Net(_p1 = cn.Net.Ecosystem[net_id], _p2 = p2, _species = self)

                else:
                    # Cloning
                    offspring = cn.Net(_p1 = cn.Net.Ecosystem[net_id], _species = self)
                    offspring.mutate(_structure = False)

                # Increase the offspring count
                Species.Offspring += 1

            else:
                cn.Net.Ecosystem[net_id].mutate()
