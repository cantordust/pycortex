import torch
torch.set_printoptions(precision = 4, threshold = 5000, edgeitems = 5, linewidth = 160)

import torch.multiprocessing as tm

from torch.nn import functional as tnf

import os
import sys
import math
from datetime import datetime
import threading as thd

from tensorboardX import SummaryWriter

import cortex.random as Rand
import cortex.statistics as Stat

from cortex.network import Net
from cortex.layer import Layer
from cortex.species import Species

# Global settings
LearningRate = 0.01
Momentum = 0.0
Epochs = 10

TrainBatchSize = 16
TestBatchSize = 1000
Runs = 1

# Data loading
Device = torch.device('cpu')
DataLoadArgs = {}

TrainFunction = None
TestFunction = None

LossFunction = tnf.cross_entropy
Optimiser = torch.optim.Adadelta

MaxThreads = None

LogDir = './logs'
DataDir = './data'
DownloadData = False

# Operating variables for saving models
ExperimentName = 'experiment'
CurrentEpoch = 1
CurrentRun = 1
LogDirPrefix = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")

Logger = None
LogInterval = 200

UnitTestMode = False

import argparse
def parse():

    # Training settings
    parser = argparse.ArgumentParser(description='PyCortex argument parser')

    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--runs', type=int, default=1, metavar='N',
                        help='number of runs (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--init-nets', type=int, default=32, metavar='N',
                        help='Initial number of networks (default: 32)')
    parser.add_argument('--max-nets', type=int, default=256, metavar='N',
                        help='Maximal number of networks (default: 256)')
    parser.add_argument('--init-species', type=int, default=8, metavar='N',
                        help='Initial number of species (default: 8)')
    parser.add_argument('--max-species', type=int, default=32, metavar='N',
                        help='Maximal number of species (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--rand-seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--max-threads', type=int, default=None, metavar='S',
                        help='number of threads (default: all available cores)')
    parser.add_argument('--experiment-name', type=str, default='Experiment', metavar='S',
                        help='Experiment name')
    parser.add_argument('--download-data', action='store_true', default=False,
                        help='Indicate whether the training data should be downloaded automatically.')
    parser.add_argument('--data-dir', type=str, default='./data', metavar='N',
                        help='Directory for storing the training / testing data')
    parser.add_argument('--log-dir', type=str, default='./logs', metavar='N',
                        help='Directory for storing the output logs')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    if args.train_batch_size:
        global TrainBatchSize
        TrainBatchSize = args.train_batch_size

    if args.test_batch_size:
        global TestBatchSize
        TestBatchSize = args.test_batch_size

    if args.runs:
        global Runs
        Runs = args.runs

    if args.epochs:
        global Epochs
        Epochs = args.epochs

    if args.init_nets:
        Net.Init.Count = args.init_nets

    if args.max_nets:
        Net.Max.Count = args.max_nets

    if args.init_species:
        Species.Init.Count = args.init_species

    if args.max_species:
        Species.Max.Count = args.max_species

    if args.lr:
        global LearningRate
        LearningRate = args.lr

    if args.momentum:
        global Momentum
        Momentum = args.momentum

    if args.cuda and torch.cuda.is_available():
        global Device
        Device = torch.device('cuda')

        global DataLoadArgs
        DataLoadArgs = {'num_workers': 1,
                        'pin_memory': True}

    if args.rand_seed is not None:
        torch.manual_seed(args.rand_seed)

    if args.max_threads is not None:
        global MaxThreads
        MaxThreads = args.max_threads

    if args.experiment_name is not None:
        global ExperimentName
        ExperimentName = args.experiment_name

    if args.log_dir:
        global LogDir
        LogDir = args.log_dir

    if args.data_dir:
        global DataDir
        DataDir = args.data_dir

    if args.download_data:
        global DownloadData
        DownloadData = args.download_data

    if args.log_interval:
        global LogInterval
        LogInterval = args.log_interval

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
        assert Species.Init.Count <= Net.Init.Count, "Initial species count (%r) is greater than the initial network count (%r)" % (Species.Init.Count, Net.Init.Count)

    else:
        Species.Init.Count = 1
        Species.Max.Count = 1

    # Generate species and nets

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

            while Species.find(proto_net.get_genome()) != 0:
                proto_net.mutate(_parameters = False)

    else:
        for net in range(net_quota):
            proto_net = Net()
            proto_net.mutate()

    print("Species:", len(Species.populations))
    for species in Species.populations.values():
        species.print()

    print("Nets:", len(Net.ecosystem))
    for net in Net.ecosystem.values():
        net.print()

    if not UnitTestMode:
        global LogDir
        LogDir += '/' + LogDirPrefix

        os.makedirs(LogDir, exist_ok = True)

        global Logger
        Logger = SummaryWriter(LogDir + '/TensorBoard')

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

    # Reset the champion.
    Net.champ = None

    # Running statistics about species fitness
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
        species.print()

def save(_net_id,
         _name = None):

    save_dir = LogDir + '/run_' + str(CurrentRun) + '/epoch_' + str(CurrentEpoch)

    os.makedirs(save_dir, exist_ok = True)

    if _name is None:
        name = 'net_' + str(_net_id)
    else:
        name = _name

    torch.save(Net.ecosystem[_net_id], save_dir + '/' + name + '.pt')

#    Net.ecosystem[_net_id].print(save_dir + '/' + name + '.txt')

def cull():

    while len(Net.ecosystem) > Net.Max.Count:

        # Get a random species ID
        species_wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)
        for species in Species.populations.values():
            species_wheel.add(species.ID, species.fitness.relative)

        species_id = species_wheel.spin()

        # Get a random network ID
        net_wheel = Rand.RouletteWheel()
        for net_id in Species.populations[species_id].nets:
            net = Net.ecosystem[net_id]
            if net.age > 0:
                net_wheel.add(net_id, net.age / Net.ecosystem[net_id].fitness.relative)

        net_id = net_wheel.spin()

        print("Erasing network ", net_id)

        # Erase the network from the species.
        Species.populations[species_id].nets.remove(net_id)

        # Erase the network from the ecosystem.
        del Net.ecosystem[net_id]

        if len(Species.populations[species_id].nets) == 0:
            del Species.populations[species_id]

def evolve(_stats):

    print("======[ Evolving ecosystem ]======")

    # Compute the relative fitness of networks and species.
    print("\t`-> Calibrating...")
    calibrate()

    if not UnitTestMode:
        for net in Net.ecosystem.values():

#            Logger.add_scalars('Stats for network ' + str(Net.ID), {
#                    'Absolute fitness': net.fitness.absolute,
#                    'Relative fitness': net.fitness.relative,
#                    'Layers': len(net.layers),
#                    'Parameters': net.get_parameter_count()
#                    },
#            CurrentEpoch)
            save(net.ID)

        Logger.add_scalar('Highest fitness', Net.ecosystem[Net.champ].fitness.absolute, CurrentEpoch)
        Logger.add_scalar('Networks', len(Net.ecosystem), CurrentEpoch)
        Logger.add_scalar('Species', len(Species.populations), CurrentEpoch)

        if Net.champ is not None:
            save(Net.champ, 'champion')

        _stats['Parameters'].update(Net.ecosystem[Net.champ].get_parameter_count())
        _stats['Accuracy'].update(Net.ecosystem[Net.champ].fitness.absolute)

        if CurrentEpoch < Epochs - 1:
            # Evolve networks in each species.
            print("\t`-> Evolving networks...")
            for species_id in list(Species.populations.keys()):
                Species.populations[species_id].evolve()

            # Eliminate unfit networks and empty species.
            print("\t`-> Culling...")
            cull()

def print_config(_file = sys.stdout,
                 _truncate = True):

    print("\n========================[ PyCortex configuration ]========================",
          "\nExperiment name:", ExperimentName,
          "\n\n======[ Network input ]======",
          "\nShape:", Net.Input.Shape,
          "\n\n======[ Network output ]======",
          "\nShape:", Net.Output.Shape,
          "\nBias:", Net.Output.Bias,
          "\nFunction:", Net.Output.Function.__name__,
          "\n\n======[ Initialisation parameters ]======",
          "\nNetwork count:", Net.Init.Count,
          "\nLayers:",
          file = _file)

    for layer_index, layer_def in enumerate(Net.Init.Layers):
        print("\tLayer %r:" % layer_index, file = _file)
        layer_def.print(_file = _file)

    print("\nFunction:", Net.Init.Function.__name__,
          "\nArguments:",
          file = _file)

    for key, val in Net.Init.Args.items():
        print("\t", key, ":", val, file = _file)

    print("\n======[ Maximal values ]======",
          "\nNetwork count:", Net.Max.Count,
          "\nNetwork age:", Net.Max.Age,
          "\n\n======[ Species ]======",
          "\nSpeciation:", "enabled" if Species.Enabled else "disabled",
          file = _file)

    if Species.Enabled:
        print("Species count:", Species.Init.Count,
              "\nMaximal count:", Species.Max.Count,
              file = _file)

    print("\n======[ Optimisation ]======",
          "\nLearning rate:", LearningRate,
          "\nMomentum:", Momentum,
          "\nEpochs:", Epochs,
          "\nTrain batch size:", TrainBatchSize,
          "\nTest batch size:", TestBatchSize,
          "\nRuns:", Runs,
          "\nDevice:", Device,
          "\nMax. threads:", MaxThreads,
              file = _file)

    if len(DataLoadArgs) > 0:
        print("\nData loader arguments:\n", file = _file)
        for key, val in DataLoadArgs:
            print("\t", key, ":", val, file = _file)
    print("Data directory:", DataDir,
          "\nDownload:", DownloadData,
          file = _file)

    print("Loss function:", LossFunction.__name__,
          "\nOptimiser:", Optimiser.__name__,
          "\nLog directory:", LogDir,
          "\nLog interval:", LogInterval,
          "\n\n=====================[ End of PyCortex configuration ]====================\n",
          file = _file)

def pause():
    key = input("Continue (Y/n)? ")
    if len(key) == 0:
        key = 'Y'
    return key

def run():

    assert TrainFunction is not None, "Please assign a function for training networks."

    global CurrentRun
    global CurrentEpoch

    stats = {
            'Parameters': Stat.SMAStat('Parameters'),
            'Accuracy': Stat.SMAStat('Accuracy')
            }

    context = tm.get_context('forkserver')

    for run in range(Runs):

        CurrentRun = run + 1

        init()

        print("===============[ Run", CurrentRun, "]===============")

        for epoch in range(Epochs):

            CurrentEpoch = epoch + 1

            print("======[ Epoch", CurrentEpoch, "]======")

            print("\t`-> Evaluating networks...")

#            with tm.Pool(processes = MaxThreads) as pool:
#                results = pool.starmap(TrainFunction, zip(Net.ecosystem.values(), [CurrentEpoch] * len(Net.ecosystem), [DataDir] * len(Net.ecosystem)))
#
#            for net in results:
#                Net.ecosystem[net.ID] = net

            ecosystem = tm.Manager().dict()
            processes = []

            # Dispatch
            for net in Net.ecosystem.values():
                processes.append(context.Process(target=TrainFunction, args=(net, CurrentEpoch, ecosystem, DataDir)))
                processes[-1].start()

            # Block until results are ready
            for process in processes:
                process.join()

            for net_id, net in ecosystem.items():
                Net.ecosystem[net_id] = net
            print("Ecosystem size:", len(Net.ecosystem))

            evolve(stats)

        for net_id in Net.ecosystem.keys():
            save(net_id)

    with open(LogDir + '/config.txt', 'w') as cfg_file:
        print_config(cfg_file)

    for stat in stats.values():
        stat.print()