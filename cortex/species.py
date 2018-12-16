# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:54:42 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

from cortex import random as Rand
from cortex.random import RouletteWheel, WeightType

class Species:

    Enabled = True

    class Init:
        Count = 8

    class Max:
        Count = 16

    # Static members
    ID = 0
    populations = {}

    @staticmethod
    def exists(_genome):
        for genome in Species.populations.values():
            if genome.matches(_genome):
                return True

        return False

    def __init__(self,
                 _genome = None,
                 _other = None,
                 _isolated = False):

        from cortex.network import Net
        from cortex.fitness import Fitness

        self.ID = 0

        # Increment the global species ID
        if not _isolated:
            Species.ID += 1
            self.ID = Species.ID
            Species.populations[self.ID] = self

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

    def matches(self,
                _other):
        """
        Equality testing for species / genome
        """

        other_genome = None
        if isinstance(_other, list):
            other_genome = _other
            #print("Comparing species", self.ID, "with a raw genome")

        elif isinstance(_other, Species):
            other_genome = _other.genome
            #print("Comparing species", self.ID, "with species", _other.ID)

        assert other_genome is not None, "Species comparison operation failed: comparing %r with %r" % (self, _other)

        if not Species.Enabled:
            # If speciation is disabled, any genome compares equal to any other genome
            return True

        #print("Genome 1 length:", len(self.genome), ", Genome 2 length:", len(other_genome))

        if len(self.genome) != len(other_genome):
            return False

        for layer_index in range(len(self.genome)):
            #print("Comparing layer ", l)
            if not self.genome[layer_index].matches(other_genome[layer_index]):
                return False

        return True

    def calibrate(self,
                  _complexity_fitness_scale):

        from cortex.network import Net
        from cortex import statistics as Stat

        if len(self.nets) == 0:
            return

        # Reset the champion
        self.champ = None

        net_stats = Stat.SMAStat()
        top_fitness = -1.0

        # Compute the absolute fitness of the species
        for net_id in self.nets:
            relative_fitness = _complexity_fitness_scale[net_id] * Net.ecosystem[net_id].fitness.absolute

            if relative_fitness > top_fitness:
                top_fitness = relative_fitness

            net_stats.update(Net.ecosystem[net_id].fitness.absolute)

        # Sort the networks in order of decreasing fitness
        self.nets = sorted(self.nets, key = lambda net_id: Net.ecosystem[net_id].fitness.absolute, reverse = True)

        if len(self.nets) > 0:
            self.champ = self.nets[0]

        print("Networks for species", self.ID, "sorted in order of descending fitness:", self.nets)
        print("Champion for species", self.ID, ":", self.champ)

        # Compute the relative fitness of the networks
        # belonging to this species
        for net_id in self.nets:
            # Compute the relative fitness
            net = Net.ecosystem[net_id]
            net.fitness.relative = net_stats.get_offset(net.fitness.absolute)

            print("Network", net_id, "fitness:",
                  "\t\tAbsolute:", Net.ecosystem[net_id].fitness.absolute,
                  "\t\tRelative:", Net.ecosystem[net_id].fitness.relative)

        self.fitness.absolute = net_stats.mean

    def evolve(self):

        from cortex.network import Net

        # Populate the parent wheel.
        # Networks that have been selected for crossover
        # will pick a partner at random by spinning the wheel.
        parent_wheel = RouletteWheel()
        for net_id in self.nets:
            parent_wheel.add(net_id, Net.ecosystem[net_id].fitness.relative)

        # Iterate over the networks and check if we should perform crossover or mutation
        for net_id in self.nets:
            if Rand.chance(Net.ecosystem[net_id].fitness.relative):
                Net(_p1 = Net.ecosystem[net_id], _p2 = Net.ecosystem[parent_wheel.spin()])

            else:
                Net.ecosystem[net_id].mutate()




