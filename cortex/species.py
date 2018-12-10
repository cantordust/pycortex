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
