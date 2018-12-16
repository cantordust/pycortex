#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:09:53 2018

@author: Alexander Hadjiivanov
@licence: MIT (https://opensource.org/licence/MIT)
"""

import os
import sys
import math
from datetime import datetime
import threading as thd
from concurrent.futures import ProcessPoolExecutor as ThreadPool

import torch
from torch.nn import functional as tnf

from cortex.network import Net
from cortex.species import Species
from cortex.random import RouletteWheel, WeightType
from cortex import statistics as Stat

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
DataLoadArgs = {}

TrainFunction = None
TestFunction = None

LossFunction = tnf.cross_entropy
Optimiser = torch.optim.Adadelta

MaxThreads = None

ArchiveDir = './model_archive'

# Operating variables for saving models
ExperimentName = 'experiment'
CurrentEpoch = 1
CurrentRun = 1
ArchiveDirPrefix = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")

print_lock = thd.Lock()

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
            proto_net.mutate()

    print("Species:", len(Species.populations))
    print("Nets:", len(Net.ecosystem))
    for species in Species.populations.values():
        print("Species", species.ID, "contains networks", *sorted(species.nets))

    global ArchiveDir
    ArchiveDir += '/' + ExperimentName + '/' + ArchiveDirPrefix

    os.makedirs(ArchiveDir, exist_ok = True)

def calibrate():

    # Remove extinct species.
    extinct = [species.ID for species in Species.populations.values() if len(species.nets) == 0]

    while len(extinct) > 0:
        del Species.populations[extinct.pop()]

    # Increase and normalise the age of all networks.
    net_stat = Stat.SMAStat()
    for net in Net.ecosystem.values():
        net.age += 1
        net_stat.update(net.age)

    # Scale the age
    for net in Net.ecosystem.values():
        net.scaled_age = net_stat.get_offset(net.age)

    # Reset the champion.
    Net.champ = None

    # A statistics package for collecting
    # species statistics on the fly.
    species_stat = Stat.SMAStat()

    # Highest fitness seen so far.
    top_fitness = -math.inf

    complexity_fitness_scale = {}
    complexity_stat = Stat.SMAStat()
    for net in Net.ecosystem.values():
        complexity_fitness_scale[net.ID] = net.get_parameter_count()
        complexity_stat.update(complexity_fitness_scale[net.ID])

    for net_id in complexity_fitness_scale.keys():
        # Lower complexity = higher scale factor.
        # This means that for identical absolute fitness, an individual with
        # lower complexity would have a higher relative fitness.
        complexity_fitness_scale[net_id] = complexity_stat.get_inv_offset(complexity_fitness_scale[net_id])
        print("Network", net_id, "fitness scale:", complexity_fitness_scale[net_id])

    for species in Species.populations.values():

        # Compute the relative network fitness
        # and the absolute species fitness.
        species.calibrate(complexity_fitness_scale)
        species_stat.update(species.fitness.absolute)

        if (Net.champ is None or
            Net.ecosystem[species.champ].fitness.absolute > top_fitness):

            Net.champ = species.champ
            top_fitness = Net.ecosystem[species.champ].fitness.absolute

    for species in Species.populations.values():
        # Compute the relative species fitness
        species.fitness.relative = species_stat.get_offset(species.fitness.absolute)

        print("Species", species.ID, "fitness:"
              "\t\tAbsolute:", species.fitness.absolute,
              "\t\tRelative:", species.fitness.relative)

def save(net_id):

    save_dir = ArchiveDir + '/run_' + str(CurrentRun) + '/epoch_' + str(CurrentEpoch)

    os.makedirs(save_dir, exist_ok = True)

    torch.save(Net.ecosystem[net_id].state_dict(), save_dir + '/net_' + str(net_id) + ".pt")

def cull():

    while len(Net.ecosystem) > Net.Max.Count:

        # Get a random species ID
        species_wheel = RouletteWheel(WeightType.Inverse)
        for species in Species.populations.values():
            species_wheel.add(species.ID, species.fitness.relative)

        species_id = species_wheel.spin()

        # Get a random network ID
        net_wheel = RouletteWheel()
        for net_id in Species.populations[species_id].nets:
            net_wheel.add(net_id, Net.ecosystem[net_id].scaled_age * (1.0 - Net.ecosystem[net_id].fitness.relative))
        net_id = net_wheel.spin()

        print("Erasing network ", net_id)

        # Erase the network from the species.
        del Species.populations[species_id].nets[net_id]

        # Archive the network
        save(net_id)

        # Erase the network from the ecosystem.
        del Net.ecosystem[net_id]

        if len(Species.populations[species_id].nets) == 0:
            del Species.populations[species_id]

def evolve():

    print(">>> Evolving ecosystem")

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

def run():

    assert TrainFunction is not None, "Please assign a function for training networks."

    global CurrentRun
    global CurrentEpoch

    for run in range(Runs):

        CurrentRun = run + 1

        init()

        for epoch in range(Epochs):

            CurrentEpoch = epoch + 1

            with ThreadPool(max_workers = MaxThreads) as threadpool:
                for net in threadpool.map(TrainFunction, Net.ecosystem.values(), [1] * len(Net.ecosystem)):
                    Net.ecosystem[net.ID] = net
                    print('Absolute fitness for network %r: %r' % (net.ID, net.fitness.absolute))

            evolve()

        for net_id in Net.ecosystem.keys():
            save(net_id)

def print_config(_file = None,
                 _truncate = True):

    if _file is None:
        fh = sys.stdout
    if isinstance(_file, str):
        fh = open(_file, 'w')
        if _truncate:
            fh.truncate()
    else:
        fh = _file

    print("\n========================[ PyCortex configuration ]========================",
          "\n======[ Network input ]======",
          "\nShape:", Net.Input.Shape,
          "\n======[ Network output ]======",
          "\nShape:", Net.Output.Shape,
          "\nBias:", Net.Output.Bias,
          "\nFunction:", Net.Output.Function.__name__,
          "\n======[ Initial values ]======",
          "\nNetwork count:", Net.Init.Count,
          "\nLayers:\n",
          file = fh)

    for layer_index, layer_def in enumerate(Net.Init.Layers):
        print("\n\tLayer", layer_index + 1, ":",
              "\n\t\tShape:", layer_def.shape,
              "\n\t\tBias:", layer_def.bias,
              "\n\t\tType:", layer_def.op.__name__,
              "\n\t\tActivation:", layer_def.activation.__name__,
              "\n\t\tConvolutional:", layer_def.is_conv,
              "\n\t\tEmpty:", layer_def.empty,
              file = fh)

    print("\nFunction:", Net.Init.Function.__name__,
          "\nArguments:",
          file = fh)

    for key, val in Net.Init.Args.items():
        print("\t", key, ":", val, file = fh)

    print("\n======[ Maximal values ]======",
          "\nNetwork count:", Net.Max.Count,
          "\nNetwork age:", Net.Max.Age,
          "\n======[ Species ]======",
          "\nSpeciation:", "enabled" if Species.Enabled else "disabled",
          file = fh)

    if Species.Enabled:
        print("\nSpecies count:", Species.Init.Count,
              "\nMaximal count:", Species.Max.Count,
              file = fh)

    print("\n======[ Optimisation ]======",
          "\nLearning rate:", LearningRate,
          "\nMomentum:", Momentum,
          "\nEpochs:", Epochs,
          "\nTrain batch size:", TrainBatchSize,
          "\nTest batch size:", TestBatchSize,
          "\nRuns:", Runs,
          "\nLog interval ( x train batch size):", LogInterval,
          "\nDevice:", Device,
              file = fh)

    if len(DataLoadArgs) > 0:
        print("\nData loader arguments:\n", file = fh)
        for key, val in DataLoadArgs:
            print("\t", key, ":", val, file = fh)

    print("\nLoss function:", LossFunction.__name__,
          "\nOptimiser:", Optimiser.__name__,
          "\nArchive directory:", ArchiveDir,
          "\n=====================[ End of PyCortex configuration ]====================\n",
          file = fh)

def pause():
    key = input("Continue (Y/n)? ")
    if len(key) == 0:
        key = 'Y'
    return key
