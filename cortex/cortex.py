import argparse
import os
import sys
import traceback
import math
import time

from datetime import datetime
from enum import IntEnum
from copy import deepcopy as dcp

from mpi4py import  MPI

import torch
torch.set_printoptions(precision = 4, threshold = 5000, edgeitems = 5, linewidth = 160)
torch.multiprocessing.set_start_method('spawn')

import torch.nn.functional as tnf

from tensorboardX import SummaryWriter

import cortex.random as Rand
import cortex.statistics as Stat
import cortex.functions as Func

import cortex.network as cn
import cortex.layer as cl
import cortex.species as cs

################[ Global variables ]################

class Tags(IntEnum):
    Ready = 0
    Start = 1
    Done = 2
    Exit = 3

class Conf:
    ExperimentName = 'Experiment'
    Runs = 1
    Epochs = 50

    TrainBatchSize = 128
    TestBatchSize = 1000

    DataDir = ''
    DownloadData = False
    DataLoadArgs = {}

    Device = torch.device('cpu')
    UseCuda = False

    LearningRate = 0.01
    Momentum = 0.5

    MaxWorkers = 1

    LogDir = './logs'
    LogInterval = 500
    Logger = None

    Optimiser = torch.optim.Adadelta
    LossFunction = tnf.cross_entropy

    OutputFunction = tnf.log_softmax
    OutputFunctionArgs = {'dim': 1}

    UnitTestMode = False

    Evaluator = None

    Workers = []
    Tag = Tags.Start

    def __init__(self):
        self.train_batch_size = Conf.TrainBatchSize
        self.test_batch_size = Conf.TestBatchSize

        self.data_dir = Conf.DataDir
        self.data_load_args = Conf.DataLoadArgs
        self.download_data = Conf.DownloadData

        self.device = Conf.Device

        self.log_interval = Conf.LogInterval

        self.optimiser = Conf.Optimiser
        self.loss_function = Conf.LossFunction

        self.output_function = Conf.OutputFunction
        self.output_function_args = Conf.OutputFunctionArgs

        self.evaluator = Conf.Evaluator

def get_rank():
    return MPI.COMM_WORLD.Get_rank()

def dump_exception():
    print("-"*60)
    traceback.print_exc()
    print("-"*60)

def init_conf():

    if get_rank() != 0:
        return

    # Training settings
    parser = argparse.ArgumentParser(description='PyCortex argument parser')

    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    parser.add_argument('--runs', type=int, help='number of runs')
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--init-nets', type=int, help='Initial number of networks')
    parser.add_argument('--max-nets', type=int, help='Maximal number of networks')
    parser.add_argument('--no-speciation', action='store_true', help='Disable speciation')
    parser.add_argument('--init-species', type=int, help='Initial number of species')
    parser.add_argument('--max-species', type=int, help='Maximal number of species')
    parser.add_argument('--train-batch-size', type=int, help='Input batch size for training')
    parser.add_argument('--test-batch-size', type=int, help='Input batch size for testing')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--momentum', type=float, help='SGD momentum')
    parser.add_argument('--use-cuda', help='Enables CUDA training')
    parser.add_argument('--rand-seed', type=int, help='Manual random seed')
    parser.add_argument('--max-workers', type=int, help='Number of workers for evaluating networks in parallel')
    parser.add_argument('--download-data', action='store_true', help='Indicate whether the training data should be downloaded automatically.')
    parser.add_argument('--data-dir', type=str, help='Directory for storing the training / testing data')
    parser.add_argument('--log-dir', type=str, help='Directory for storing the output logs')
    parser.add_argument('--log-interval', type=int, help='Interval between successive log outputs (in untis of training batch size)')

    args = parser.parse_args()

    # CLI arguments
    if args.experiment_name:
        Conf.ExperimentName = args.experiment_name

    if args.runs:
        Conf.Runs = args.runs

    if args.epochs:
        Conf.Epochs = args.epochs

    if args.init_nets:
        cn.Net.Init.Count = args.init_nets

    if args.max_nets:
        cn.Net.Max.Count = args.max_nets

    if args.no_speciation:
        cs.Species.Enabled = False

    if args.init_species:
        cs.Species.Init.Count = args.init_species

    if args.max_species:
        cs.Species.Max.Count = args.max_species

    if args.train_batch_size:
        Conf.TrainBatchSize = args.train_batch_size

    if args.test_batch_size:
        Conf.TestBatchSize = args.test_batch_size

    if args.data_dir:
        Conf.DataDir = args.data_dir

    if args.download_data:
        Conf.DownloadData = args.download_data

    if args.use_cuda and torch.cuda.is_available():
        Conf.Device = torch.device('cuda')
        Conf.UseCuda = True

        Conf.DataLoadArgs['num_workers'] = 1
        Conf.DataLoadArgs['pin_memory'] = True

    if args.learning_rate:
        Conf.LearningRate = args.learning_rate

    if args.momentum:
        Conf.Momentum = args.momentum

    if args.max_workers:
        Conf.MaxWorkers = args.max_workers

    if args.log_dir:
        Conf.LogDir = args.log_dir

    Conf.LogDir = Conf.LogDir + '/' + datetime.now().strftime("%d_%b_%Y_%H_%M_%S")

    if args.log_interval:
        Conf.LogInterval = args.log_interval

    if args.rand_seed is not None:
        torch.manual_seed(args.rand_seed)

def print_conf(_file = sys.stdout):

    if get_rank() != 0:
        return

    str = f'''
========================[ PyCortex configuration ]========================
>>> Experiment name: {Conf.ExperimentName}
>>> Runs: {Conf.Runs}
>>> Epochs: Conf.Epochs
>>> Init. networks: cn.Net.Init.Count
>>> Max. networks: cn.Net.Max.Count
>>> Speciation: {"enabled" if cs.Species.Enabled else "disabled"}'''

    if cs.Species.Enabled:
        str += f'''
    Init. species: {cs.Species.Init.Count}
    Max. species: {cs.Species.Max.Count}'''

    str += f'''
>>> Learning rate: {Conf.LearningRate}
>>> Momentum: {Conf.Momentum}
>>> CUDA: {"enabled" if Conf.UseCuda else "disabled"}
>>> Input shape: {cn.Net.Input.Shape}
>>> Output shape: {cn.Net.Output.Shape}
>>> Layer bias: {cl.Layer.Bias}
>>> Layer activations:'''

    for key, val in cl.Layer.Activations.items():
        str += f'''
    {key}: {val.__name__}'''

    for layer_index, layer_def in enumerate(cn.Net.Init.Layers):
        str += layer_def.as_str()

    str += f'''
>>> Init. function: {cl.Layer.InitFunction.__name__}
>>> Init. arguments:'''

    for key, val in cl.Layer.InitArgs.items():
        str += f'''
    {key}: {val}'''

    str += f'''
>>> Max. nets: {cn.Net.Max.Count}
>>> Max. net age: {cn.Net.Max.Age}
>>> Learning rate: {Conf.LearningRate}
>>> Momentum: {Conf.Momentum}
>>> Device: {Conf.Device}
>>> Max. workers: {Conf.MaxWorkers}'''


    str += '>>> Data loader arguments:'
    for key, val in Conf.DataLoadArgs.items():
        str += f'''
    {key}: {val}'''

    str += f'''
>>> Data directory: {Conf.DataDir}
>>> Download: {Conf.DownloadData}
>>> Optimiser: {Conf.Optimiser.__name__}
>>> Loss function: {Conf.LossFunction.__name__}
>>> Log directory: {Conf.LogDir}
>>> Log interval: {Conf.LogInterval}
>>> Unit test mode: {Conf.UnitTestMode}
=====================[ End of PyCortex configuration ]===================='''

    print(str, file = _file)

def pause():
    key = input("Continue (Y/n)? ")
    if len(key) == 0:
        key = 'Y'
    return key

def init():

    if get_rank() != 0:
        return

    """
    Initialise the ecosystem and generate initial species if speciation is enabled.
    """

    print(">>> Initialising ecosystem...")

    # Reset the static network attributes
    cn.Net.reset()

    # Reset the static species attributes
    cs.Species.reset()

    # Sanity check on the species count
    if cs.Species.Enabled:
        assert cs.Species.Init.Count > 0, "Invalid initial species count %r" % cs.Species.Init.Count
        assert cs.Species.Init.Count <= cn.Net.Init.Count, "Initial species count (%r) is greater than the initial network count (%r)" % (cs.Species.Init.Count, cn.Net.Init.Count)

    else:
        cs.Species.Init.Count = 1
        cs.Species.Max.Count = 1

    # Population size (the number of networks per species).
    net_quota = cn.Net.Init.Count // cs.Species.Init.Count

    # Initial proto-net.
    # This is the first self-replicating prion to spring
    # into existence in the digital primordial bouillon.
    proto_net = cn.Net(_isolated = True)

    probabilities = {
                'layer': 1,
                'node': 1,
                'stride': 1,
                'kernel': 1
                }

    if cs.Species.Enabled:
        # Generate proto-species and proto-nets
        while True:

            # Generate proto-species
            proto_species = cs.Species(_genome = proto_net.get_genome())

            # Generate proto-nets for this proto-species.
            for n in range(net_quota):
                proto_net = cn.Net(_species = proto_species)

            if len(cs.Species.Populations) == cs.Species.Init.Count:
                break

            proto_net = cn.Net(_species = proto_species, _isolated = True)

            while cs.Species.find(proto_net.get_genome()) != 0:
                proto_net.mutate(_probabilities = probabilities, _complexify = True)

    else:

        # Generate proto-species
        proto_species = cs.Species(_genome = proto_net.get_genome())

        # Generate proto-nets.
        for n in range(net_quota):
            proto_net = cn.Net(_species = proto_species)
            proto_net.mutate(_parameters = False)

    print(f'Network count: {len(cn.Net.Ecosystem)}')
    for net in cn.Net.Ecosystem.values():
        print(net.as_str())

    print(f'Species count: {len(cs.Species.Populations)}')
    for species in cs.Species.Populations.values():
        species.print()

    if not Conf.UnitTestMode:
        os.makedirs(Conf.LogDir, exist_ok = True)
        Conf.Logger = SummaryWriter(Conf.LogDir + '/TensorBoard')

def calibrate():

    if get_rank() != 0:
        return

    # Remove extinct species.
    extinct = [species.ID for species in cs.Species.Populations.values() if len(species.nets) == 0]

    while len(extinct) > 0:
        del cs.Species.Populations[extinct.pop()]

    # Increase and normalise the age of all networks.
    net_stat = Stat.SMAStat()
    for net in cn.Net.Ecosystem.values():
        net.age += 1
        net_stat.update(net.age)

    # Reset the global champion.
    cn.Net.Champion = None

    # Running statistics about species fitness
    species_stat = Stat.SMAStat()

    # Highest fitness seen so far.
    top_fitness = -math.inf

    complexity_fitness_scale = {}
    complexity_stat = Stat.SMAStat()
    for net in cn.Net.Ecosystem.values():
        complexity_fitness_scale[net.ID] = net.get_parameter_count()
        complexity_stat.update(complexity_fitness_scale[net.ID])

    for net_id in complexity_fitness_scale.keys():
        # Lower complexity = higher scale factor.
        # This means that for identical absolute fitness, an individual with
        # lower complexity would have a higher relative fitness.
        complexity_fitness_scale[net_id] = complexity_stat.get_inv_offset(complexity_fitness_scale[net_id])
#        print("Network", net_id, "fitness scale:", complexity_fitness_scale[net_id])

    for species in cs.Species.Populations.values():

        # Compute the relative network fitness
        # and the absolute species fitness.
        species.calibrate(complexity_fitness_scale)
        species_stat.update(species.fitness.absolute)

        if (cn.Net.Champion is None or
            cn.Net.Ecosystem[species.champion].fitness.absolute > top_fitness):

            cn.Net.Champion = species.champion
            top_fitness = cn.Net.Ecosystem[species.champion].fitness.absolute

    for species in cs.Species.Populations.values():
        # Compute the relative species fitness
        species.fitness.relative = species_stat.get_offset(species.fitness.absolute)
#        species.print()

    print(f'>>> Global champion: {cn.Net.Champion} (fitness: {cn.Net.Ecosystem[cn.Net.Champion].fitness.absolute})')

def save(_net_id,
         _run,
         _epoch,
         _name = None):

    if get_rank() != 0:
         return

    save_dir = Conf.LogDir + '/run_' + str(_run) + '/epoch_' + str(_epoch)

    os.makedirs(save_dir, exist_ok = True)

    if _name is None:
        name = 'net_' + str(_net_id)
    else:
        name = _name

    torch.save(cn.Net.Ecosystem[_net_id], save_dir + '/' + name + '.pt')

    with open(save_dir + '/' + name + '.txt', 'w+') as plaintext:
        print(cn.Net.Ecosystem[_net_id].as_str(), file = plaintext)

def cull():

    if get_rank() != 0:
        return

    wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)

    for net_id, net in cn.Net.Ecosystem.items():
        if (net.age > 0 and
            net_id != cn.Net.Champion):
            # Networks whose fitness is low or hasn't changed much for a while
            # are more likely to be eliminated.
            wheel.add(net_id, net.fitness.relative * net.fitness.stat.get_sd() / net.age)

    removed_nets = []
    removed_species = []
    while len(cn.Net.Ecosystem) > cn.Net.Max.Count:

        # Get a random network ID
        net_id = wheel.pop()

        if net_id is not None:

            species_id = cn.Net.Ecosystem[net_id].species_id

#            print('Removing network {} from species {}'.format(net_id, species_id))
            removed_nets.append(net_id)

            # Remove the network from the species.
            cs.Species.Populations[species_id].nets.remove(net_id)

            # Remove the network from the ecosystem.
            del cn.Net.Ecosystem[net_id]

        if len(cs.Species.Populations[species_id].nets) == 0:
            removed_species.append(species_id)
            del cs.Species.Populations[species_id]

    print('Removed nets: {}'.format(removed_nets))
    print('Removed species: {}'.format(removed_species))

def evolve(_stats,
           _run,
           _epoch):

    if get_rank() != 0:
       return

    print("======[ Evolving ecosystem ]======")

    # Compute the relative fitness of networks and species.
    print("\t`-> Calibrating...")
    calibrate()

    # Set the offspring count to 0
    cs.Species.Offspring = 0

    if not Conf.UnitTestMode:

        #        for net in cn.Net.Ecosystem.values():
#            Conf.Logger.add_scalars('Stats for network ' + str(net.ID), {
#                                    'Absolute fitness': net.fitness.absolute,
#                                    'Relative fitness': net.fitness.relative,
#                                    'Layers': len(net.layers),
#                                    'Parameters': net.get_parameter_count()
#                                    },
#                        _epoch)
#            save(net.ID, _run, _epoch)

        Conf.Logger.add_scalar('Highest fitness', cn.Net.Ecosystem[cn.Net.Champion].fitness.absolute, _epoch)
        Conf.Logger.add_scalar('Average fitness', _stats['Accuracy'].mean, _epoch)
        Conf.Logger.add_scalar('Parameter count for champion', cn.Net.Ecosystem[cn.Net.Champion].get_parameter_count(), _epoch)
        Conf.Logger.add_scalar('Network count', len(cn.Net.Ecosystem), _epoch)
        Conf.Logger.add_scalar('Species count', len(cs.Species.Populations), _epoch)

        if cn.Net.Champion is not None:
            save(cn.Net.Champion, _run, _epoch, 'champion')

        _stats['Parameters'].update(cn.Net.Ecosystem[cn.Net.Champion].get_parameter_count())
        _stats['Accuracy'].update(cn.Net.Ecosystem[cn.Net.Champion].fitness.absolute)

        if _epoch < Conf.Epochs:
            # Evolve networks in all species.
            print("\t`-> Evolving networks...")

            wheel = Rand.RouletteWheel()

            for species_id, species in cs.Species.Populations.items():
                wheel.add(species_id, species.fitness.relative * species.fitness.stat.get_sd())

            while not wheel.is_empty():
                cs.Species.Populations[wheel.pop()].evolve()

            # Eliminate unfit networks and empty species.
            if len(cn.Net.Ecosystem) > cn.Net.Max.Count:
                print("\t`-> Culling...")
                cull()

def run():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    if rank == 0:

        # Master process

        assert Conf.Evaluator is not None, "Please assign a function for training networks."

        # Wait for all workers to return a Ready signal

        while len(Conf.Workers) < size - 1:
            while not comm.iprobe(source=MPI.ANY_SOURCE):
                pass

            comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == Tags.Ready:
                print('Worker {} ready'.format(source))
                Conf.Workers.append(source)

            elif tag == Tags.Exit:
                Conf.Tag = Tags.Exit
                break

        # List of free workers
        free_workers = []
        for worker in Conf.Workers:
            free_workers.append(worker)

        # Experiment statistics
        stats = {
                'Parameters': Stat.SMAStat('Parameters'),
                'Accuracy': Stat.SMAStat('Accuracy')
                }

        # Worker management subroutine
        def manage_workers(free_workers):

            while not comm.iprobe(source=MPI.ANY_SOURCE):
                time.sleep(0.1)

            net = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == Tags.Done:
                cn.Net.Ecosystem[net.ID] = net
                free_workers.append(source)

            elif tag == Tags.Exit:
                Conf.Tag = Tags.Exit
                Conf.Workers.remove(source)

        # Run the task
        for run in range(1, Conf.Runs + 1):

            if len(Conf.Workers) == 0:
                break

            print("===============[ Run", run, "]===============")

            # Fresh configuration
            conf = Conf()

            conf.run = run

            # Initialise the ecosystem
            try:
                init()

            except:
                print('Caught exception in init()')
                Conf.Tag = Tags.Exit
                dump_exception()

            for epoch in range(1, Conf.Epochs + 1):

                if len(Conf.Workers) == 0:
                    break

                print("======[ Epoch", epoch, "]======")

                conf.epoch = epoch

                print("\t`-> Evaluating networks...")

                # Dispatch the networks for evaluation

                net_ids = dcp(list(cn.Net.Ecosystem.keys()))

                while (len(Conf.Workers) > 0 and
                       len(net_ids) > 0):

                    # Wait for a free worker
                    while (len(Conf.Workers) > 0 and
                           len(free_workers) == 0):
                        manage_workers(free_workers)

                    if len(free_workers) > 0:
                        worker = free_workers.pop()
                        net_id = net_ids.pop()
                        package = (cn.Net.Ecosystem[net_id], conf)
                        comm.send(package, dest = worker, tag=Conf.Tag)

                # Wait for the last workers to finish
                while len(free_workers) < len(Conf.Workers):
                    manage_workers(free_workers)

                if len(Conf.Workers) > 0:
                    try:
                        evolve(stats, run, epoch)

                    except:
                        print('Caught exception in evolve()')
                        Conf.Tag = Tags.Exit
                        dump_exception()

#            for net_id in cn.Net.Ecosystem.keys():
#                save(net_id, run, epoch)

        with open(Conf.LogDir + '/config.txt', 'w+') as cfg_file:
            print_conf(_file = cfg_file)

        for key, stat in stats.items():
            stat.print()
            with open(Conf.LogDir + '/' + str(key) + '_stat.txt', 'w+') as stat_file:
                stat.print(_file = stat_file)

        for worker in Conf.Workers:
            comm.send(None, dest = worker, tag=Tags.Exit)

    else:

        comm.send(None, dest=0, tag=Tags.Ready)

        while True:

            while not comm.iprobe(source=0):
                time.sleep(0.1)

            package = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == Tags.Start:

                net, conf = package[0], package[1]

                try:

                    conf.evaluator(conf, net)
                    comm.send(net, dest=0, tag = Tags.Done)

                except Exception:
                    print('Caught exception in worker {}'.format(rank))
                    dump_exception()
                    break

            elif tag == Tags.Exit:
                break

        comm.send(None, dest=0, tag=Tags.Exit)

    print('Worker {} exiting...'.format(rank))
