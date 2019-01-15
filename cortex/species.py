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

    def as_str(self):

        str = f'\n===============[ Species {self.ID} ]===============' +\
              f'\nAbsolute fitness: {self.fitness.absolute}' +\
              f'\nRelative fitness: {self.fitness.relative}' +\
              f'\nNetworks: {self.nets}' +\
              f'\nChampion: {self.champion}'

        for layer_index, layer in enumerate(self.genome):
            str += f'\n\n======[ Layer {layer_index} ]======\n{layer.as_str()}'

        return str

    def calibrate(self):

        if len(self.nets) == 0:
            return

        import cortex.network as cn
        import cortex.functions as Func

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

        print(f'>>> Champion for species {self.ID}: {self.champion} (fitness: {cn.Net.Ecosystem[self.champion].fitness.absolute})')

        # Compute the relative fitness of the networks
        # belonging to this species
        for net_id in self.nets:
            # Compute the relative fitness
            net = cn.Net.Ecosystem[net_id]
            net.fitness.relative = net_stats.get_offset(net.fitness.absolute)

#            print("Network", net_id, "fitness:",
#                  "\t\tAbsolute:", cn.Net.Ecosystem[net_id].fitness.absolute,
#                  "\t\tRelative:", cn.Net.Ecosystem[net_id].fitness.relative)

        self.fitness.set(net_stats.mean)

    def evolve(self):

        import cortex.network as cn

        # Populate the parent wheel.
        # Networks that have been selected for crossover
        # will pick a partner at random by spinning the wheel.
        parent1_wheel = Rand.RouletteWheel()
        parent2_wheel = Rand.RouletteWheel()
        for net_id in self.nets:
            if cn.Net.Ecosystem[net_id].age > 0:
                parent1_wheel.add(net_id, cn.Net.Ecosystem[net_id].fitness.relative)
                parent2_wheel.add(net_id, cn.Net.Ecosystem[net_id].fitness.relative)

        # Iterate over the networks and check if we should perform crossover or mutation
        while not parent1_wheel.is_empty():

            # Choose one parent
            p1 = cn.Net.Ecosystem[parent1_wheel.pop()]

            # Choose a partner
            p2 = cn.Net.Ecosystem[parent2_wheel.spin()]

            # Fitter networks have a better chance of mating.
            # Take the average of the two parents' relative fitness values
            mating_chance = 0.5 * (p1.fitness.relative + p2.fitness.relative) / (Species.Offspring + 1)
            if not Species.Enabled:
                mating_chance *= p1.genome_overlap(p2)

            print(f'Chance of mating nets {p1.ID} and {p2.ID}: {mating_chance}')

            if (Species.Offspring < len(cn.Net.Ecosystem) // 2 and
                Rand.chance(mating_chance)):

                if p1 != p2:
                    # Crossover
                    offspring = cn.Net(_p1 = p1, _p2 = p2)

                else:
                    # Clone
                    offspring = cn.Net(_p1 = p1)
                    offspring.mutate(_structure = False)

                # Increase the offspring count
                Species.Offspring += 1

            elif (p1 != self.champion and
                  Rand.chance(1.0 / (p1.age + 1))):

                probabilities = {
                                'layer': 1,
                                'node': 1,
                                'stride': 1,
                                'kernel': 1
                                }

                p1.mutate(_probabilities = probabilities)

                if p1.species_id != self.ID:
                    # The network has moved to another species.
                    # Remove it from the other wheel.
                    parent2_wheel.remove(p1.ID)


