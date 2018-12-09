#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:09:53 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

#import conf as mConf
from .network import Net
from .species import Species

####[ Global settings ]####

LearningRate = 0.01
Momentum = 0.0
Epochs = 5
BatchSize = 4
Runs = 1
Threads = 4


def pause():
    input("Press <ENTER> to continue...")

def init():
    print(">>> Initialising environment...")

    # Reset the static network attributes
    Net.champ = None
    Net.population.clear()
    Net.ID = 0

    # Reset the static species attributes
    Species.env.clear()
    Species.ID = 0

    # Generate speciess and nets

    # Population size (the number of networks per species).
    net_quota = int(Net.Init.Count)

    # If speciation is enabled, distribute the net quota
    # evenly among the initial species.
    if (Species.Init.Count > 0 and
        Species.Max.Count > 0):

        net_quota = net_quota // Species.Init.Count

        # Proto-net.
        proto_net = Net(_isolated = True)
        
        # Generate proto-speciess and proto-nets
        while True:

            # Generate proto-speciess and proto-nets
            proto_species = Species(_genome = proto_net.get_genome())

            # Add the proto-nets for the species.
            for n in range(net_quota):
                proto_net = Net(_species = proto_species)
            
            if len(Species.env) == Species.Init.Count:
                break

            proto_net = Net(_species = proto_species, _isolated = True)

            while Species.exists(proto_net.get_genome()):
                proto_net.mutate(_parameters = False)

    else:
        for net in range(net_quota):
            proto_net = Net()
            
    print("Species:", len(Species.env))
    print("Nets:", len(Net.population))
    for species in Species.env.values():
        print("Species", species.ID, "contains networks", *sorted(species.nets))

def evolve():

    print(">>> Evolving environment")

    # Increase the age of all networks.
    for net in Net.population:
        net.age += 1

    # Reset the champion.
    Net.champ = None

    # Compute the relative fitness of networks and speciess.
    print("\t`-> Calibrating...")
    calibrate()

    # Evolve networks in each species.
    print("\t`-> Evolving networks...")
    for species in Species.env.values():
        species.evolve()

    # Eliminate unfit networks and empty speciess.
    print("\t`-> Culling...")
    cull()

def calibrate():

    from . import statistics as Stat
    
    # A statistics package for collecting
    # species statistics on the fly.
    species_stat = Stat.SMAStat()

    # Highest fitness seen so far.
    top_fitness = -1

    for species in Species.env:

        # Compute the relative network fitness
        # and the absolute species fitness.
        (species_fitness, species_champ) = species.calibrate()

        if (Net.champ is None or
            (species_champ is not None and Net.population[species_champ].fitness.absolute.value > top_fitness)):

            Net.champ = species_champ
            top_fitness = Net.population[species_champ].fitness.absolute

    for species in Species.env:
        # Compute the relative species fitness
        species.fitness.calibrate(species_stat)

        print("Species ", species.ID, "fitness:"
             "\t\tAbsolute: ", species.fitness.absolute.value, "(mean", species.fitness.absolute.mean, ", sd", species.fitness.absolute.sd() + ")"
             "\t\tRelative: ", species.fitness.relative.value, "(mean", species.fitness.relative.mean, ", sd", species.fitness.relative.sd() + ")")

def cull():

    from . import rand as Rand

    while len(Net.population) > Net.Max.Count:

        # Get a random species ID
        species_wheel = [(1.0 - species.fitness.relative.value) for species in Species.env.values()]
        species_id = Rand.roulette(Species.env.keys(), species_wheel)

        # Get a random network ID
        net_wheel = [Net.population[net_id].age * (1.0 - Net.population[net_id].fitness.relative.value) for net_id in Species.env[species_id].nets]
        net_id = Rand.roulette(Species.env[species_id].nets.keys(), net_wheel)

        print("Erasing network ", net_id)

        # Erase the network from the species.
        del Species.env[species_id].nets[net_id]

        # Erase the network from the environment.
        del Net.population[net_id]

    # Remove extinct species.
    extinct = [species.ID for species in Species.env.values() if len(species.nets) == 0]

    while len(extinct) > 0:
        del Species.env[extinct.pop()]
