#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:09:53 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

import torch
from torch.nn import functional as tnf

from cortex.network import Net
from cortex.species import Species
from cortex.layer import Layer
from cortex import random as Rand

# Global settings
LearningRate = 0.01
Momentum = 0.0
Epochs = 10
TrainBatchSize = 8
TestBatchSize = 1000
Runs = 1
LogInterval = 10

# Data loading
Device = torch.device('cpu')
TrainDataLoadFunc = None
TestDataLoadFunc = None
DataLoadArgs = {}

LossFunction = tnf.cross_entropy
Optimiser = torch.optim.Adadelta

def init():
    """
    Initialise the ecosystem and generate initial species if speciation is enabled.
    """

    print(">>> Initialising ecosystem...")

    # Reset the static network attributes
    Net.champ = None
    Net.ecosystem.clear()
    Net.ID = 0

    # Reset the static species attributes
    Species.populations.clear()
    Species.ID = 0

    # Sanity check on the species count
    if Species.Enabled:
        assert Species.Init.Count > 0, "Invalid initial species count %r" % Species.Init.Count
        assert Species.Init.Count < Net.Init.Count, "Initial species count (%r) is greater than the initial network count (%r)" % (Species.Init.Count, Net.Init.Count)

    else:
        Species.Init.Count = 1
        Species.Max.Count = 1

    # Generate speciess and nets

    # Population size (the number of networks per species).
    net_quota = int(Net.Init.Count)

    # If speciation is enabled, distribute the net quota
    # evenly among the initial species.
    if Species.Enabled:

        net_quota = net_quota // Species.Init.Count

        # Initial proto-net.
        # This is the first self-replicating prion to spring
        # into existence in the digital primordial bouillon.
        proto_net = Net(_isolated = True)

        # Generate proto-species and proto-nets
        while True:

            # Generate proto-species
            proto_species = Species(_genome = proto_net.get_genome())

            # Generate proto-nets for this proto-species.
            for n in range(net_quota):
                proto_net = Net(_species = proto_species)

            if len(Species.populations) == Species.Init.Count:
                break

            proto_net = Net(_species = proto_species, _isolated = True)

            while Species.exists(proto_net.get_genome()):
                proto_net.mutate(_parameters = False)

    else:
        for net in range(net_quota):
            proto_net = Net()

    print("Species:", len(Species.populations))
    print("Nets:", len(Net.ecosystem))
    for species in Species.populations.values():
        print("Species", species.ID, "contains networks", *sorted(species.nets))

def calibrate():

    from cortex import statistics as Stat

    # A statistics package for collecting
    # species statistics on the fly.
    species_stat = Stat.SMAStat()

    # Highest fitness seen so far.
    top_fitness = -1

    for species in Species.populations:

        # Compute the relative network fitness
        # and the absolute species fitness.
        (species_fitness, species_champ) = species.calibrate()

        if (Net.champ is None or
            (species_champ is not None and Net.ecosystem[species_champ].fitness.absolute.value > top_fitness)):

            Net.champ = species_champ
            top_fitness = Net.ecosystem[species_champ].fitness.absolute

    for species in Species.populations:
        # Compute the relative species fitness
        species.fitness.calibrate(species_stat)

        print("Species ", species.ID, "fitness:"
             "\t\tAbsolute: ", species.fitness.absolute.value, "(mean", species.fitness.absolute.mean, ", sd", species.fitness.absolute.sd() + ")"
             "\t\tRelative: ", species.fitness.relative.value, "(mean", species.fitness.relative.mean, ", sd", species.fitness.relative.sd() + ")")

def cull():

    from cortex import random as Rand

    while len(Net.ecosystem) > Net.Max.Count:

        # Get a random species ID
        species_wheel = [(1.0 - species.fitness.relative.value) for species in Species.populations.values()]
        species_id = Rand.roulette(Species.populations.keys(), species_wheel)

        # Get a random network ID
        net_wheel = [Net.ecosystem[net_id].age * (1.0 - Net.ecosystem[net_id].fitness.relative.value) for net_id in Species.populations[species_id].nets]
        net_id = Rand.roulette(Species.populations[species_id].nets.keys(), net_wheel)

        print("Erasing network ", net_id)

        # Erase the network from the species.
        del Species.populations[species_id].nets[net_id]

        # Erase the network from the ecosystem.
        del Net.ecosystem[net_id]

    # Remove extinct species.
    extinct = [species.ID for species in Species.populations.values() if len(species.nets) == 0]

    while len(extinct) > 0:
        del Species.populations[extinct.pop()]

def evolve():

    print(">>> Evolving ecosystem")

    # Increase the age of all networks.
    for net in Net.ecosystem:
        net.age += 1

    # Reset the champion.
    Net.champ = None

    # Compute the relative fitness of networks and speciess.
    print("\t`-> Calibrating...")
    calibrate()

    # Evolve networks in each species.
    print("\t`-> Evolving networks...")
    for species in Species.populations.values():
        species.evolve()

    # Eliminate unfit networks and empty speciess.
    print("\t`-> Culling...")
    cull()

def print_config():

    print("\n========================[ PyCortex configuration ]========================")

    print("\n======[ Network input ]======")
    print("Shape:", Net.Input.Shape)

    print("\n======[ Network output ]======")
    print("Shape:", Net.Output.Shape)
    print("Bias:", Net.Output.Bias)
    print("Function:", Net.Output.Function.__name__)

    print("\n======[ Initial values ]======")
    print("Network count:", Net.Init.Count)
    print("Layers:\n")
    for layer_index, layer_def in enumerate(Net.Init.Layers):
        print("\tLayer", layer_index + 1, ":")
        print("\t\tShape:", layer_def.shape)
        print("\t\tBias:", layer_def.bias)
        print("\t\tType:", layer_def.op.__name__)
        print("\t\tActivation:", layer_def.activation.__name__)
        print("\t\tConvolutional:", layer_def.is_conv)
        print("\t\tEmpty:", layer_def.empty)

    print("\nFunction:", Net.Init.Function.__name__)
    print("Arguments:")
    for key, val in Net.Init.Args.items():
        print("\t", key, ":", val)

    print("\n======[ Maximal values ]======")
    print("Network count:", Net.Max.Count)
    print("Network age:", Net.Max.Age)

    print("\n======[ Species ]======")
    print("Speciation:", "enabled" if Species.Enabled else "disabled")
    if Species.Enabled:
        print("Species count:", Species.Init.Count)
        print("Maximal count:", Species.Max.Count)

    print("\n======[ Optimisation ]======")
    print("Learning rate:", LearningRate)
    print("Momentum:", Momentum)
    print("Epochs:", Epochs)
    print("Train batch size:", TrainBatchSize)
    print("Test batch size:", TestBatchSize)
    print("Runs:", Runs)
    print("Log interval ( x train batch size):", LogInterval)

    print("Device:", Device)
    if len(DataLoadArgs) > 0:
        print("Data loader arguments:\n")
        for key, val in DataLoadArgs:
            print("\t", key, ":", val)

    print("Loss function:", LossFunction.__name__)
    print("Optimiser:", Optimiser.__name__)

    print("\n=====================[ End of PyCortex configuration ]====================\n")

def pause():
    key = input("Continue (Y/n)? ")
    if len(key) == 0:
        key = 'Y'
    return key
