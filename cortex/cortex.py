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
import torchvision

from tensorboardX import SummaryWriter

import cortex.random as Rand
import cortex.statistics as Stat
import cortex.functions as Func
import cortex.containers as Cont

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
    Episodes = 5

    DiscountFactor = 0.99
    Epsilon = 1.0e-8

    Device = torch.device('cpu')
    UseCuda = False
    GPUCount = 0
    MaxWorkers = 1

    DataDir = ''
    DownloadData = False
    DataLoadArgs = {}
    DataLoader = None

    TrainBatchSize = 128
    TestBatchSize = 1000
    TrainPortion = None

    LogDir = './logs'
    LogInterval = 500
    Logger = None

    OutputFunction = tnf.log_softmax
    OutputFunctionArgs = {'dim': 1}

    LossFunction = tnf.nll_loss

    Optimiser = torch.optim.Adadelta
    OptimiserArgs = {
                'lr': 1.0
                }

    UnitTestMode = False

    Evaluator = None

    GPUs = []
    Workers = []
    Tag = Tags.Start

    Stats = {
            'Species_count': Stat.SMAStat(),
            'Network_count': Stat.SMAStat(),
            'Champion_parameter_count': Stat.SMAStat(),
            'Highest_fitness': Stat.SMAStat()
            }

    def __init__(self,
                 _run,
                 _epoch,
                 _gpu_slot = None):
        self.run = _run
        self.epoch = _epoch
        self.episodes = Conf.Episodes
        self.gpu_slot = None

        self.train_batch_size = Conf.TrainBatchSize
        self.test_batch_size = Conf.TestBatchSize
        self.train_portion = Conf.TrainPortion

        self.data_dir = Conf.DataDir
        self.data_load_args = Conf.DataLoadArgs
        self.download_data = Conf.DownloadData

        self.device = Conf.Device

        self.log_interval = Conf.LogInterval

        self.loss_function = Conf.LossFunction

        self.optimiser = Conf.Optimiser
        self.optimiser_args = Conf.OptimiserArgs

        self.output_function = Conf.OutputFunction
        self.output_function_args = Conf.OutputFunctionArgs

        self.evaluator = Conf.Evaluator
        self.data_loader = Conf.DataLoader

        self.discount_factor = Conf.DiscountFactor
        self.epsilon = Conf.Epsilon

def get_rank():
    return MPI.COMM_WORLD.Get_rank()

def dump_exception():
    print("-"*60)
    traceback.print_exc()
    print("-"*60)

def save_stat(_stat,
              _path = ''):

    _path = Conf.LogDir + _path
    _path.replace('//', '/')

    os.makedirs(_path, exist_ok = True)

    with open(_path + '/' + _stat.title + '_stats.txt', 'w+') as stat_file:
        print(_stat.as_str(), file = stat_file)

def save_net(_net,
             _path,
             _name = ''):

    if get_rank() != 0:
         return

    _path = Conf.LogDir + _path
    _path.replace('//', '/')

    os.makedirs(_path, exist_ok = True)

    if _name is None:
        _name = 'net_' + str(_net.ID)

    torch.save(_net, _path + '/' + _name + '.pt')

    with open(_path + '/' + _name + '.txt', 'w+') as plaintext:
        print(_net.as_str(_parameters = True), file = plaintext)

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
    parser.add_argument('--train-portion', type=float, help='Input batch size for training')
    parser.add_argument('--test-batch-size', type=int, help='Input batch size for testing')
    parser.add_argument('--use-cuda', action='store_true', help='Enables CUDA training')
    parser.add_argument('--gpu-count', type=int, help='Indicate how many GPUs are available')
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

    if args.train_portion:
        Conf.TrainPortion = args.train_portion

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

    if args.gpu_count:
        Conf.GPUCount = args.gpu_count

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

    str = f'========================[ PyCortex configuration ]========================' + \
          f'\n>>> Experiment name: {Conf.ExperimentName}' + \
          f'\n>>> Runs: {Conf.Runs}' + \
          f'\n>>> Epochs: {Conf.Epochs}' + \
          f'\n>>> Init. networks: {cn.Net.Init.Count}' + \
          f'\n>>> Max. networks: {cn.Net.Max.Count}' + \
          f'\n>>> Max. net age: {cn.Net.Max.Age}' + \
          f'\n>>> Speciation: {"enabled" if cs.Species.Enabled else "disabled"}'

    if cs.Species.Enabled:
        str += f'\n\tInit. species: {cs.Species.Init.Count}' + \
               f'\n\tMax. species: {cs.Species.Max.Count}'

    str += f'\n>>> Discount factor: {Conf.DiscountFactor}' + \
           f'\n>>> Input shape: {cn.Net.Input.Shape}' + \
           f'\n>>> Output shape: {cn.Net.Output.Shape}' + \
           f'\n>>> Layer bias: {cl.Layer.Bias}' + \
           f'\n>>> FC layer recurrence: {"enabled" if cl.Layer.RecurrentFC else "disabled"}' + \
           f'\n>>> Layer activations:'

    for key, val in cl.Layer.Activations.items():
        str += f'\n\t{key}: {val.__name__ if val is not None else None}'

    for layer_index, layer_def in enumerate(cn.Net.Init.Layers):
        title = f'\n========[ Initial layer {layer_index} ]========'
        str += title + layer_def.as_str() + '\n' + ''.join(['=' for chars in range(len(title) - 1)])

    str += f'\n>>> Init. function: {cl.Layer.InitFunction.__name__}' + \
           f'\n>>> Init. arguments:'

    for key, val in cl.Layer.InitArgs.items():
          str += f'\n\t{key}: {val}'

    str += f'\n>>> Device: {Conf.Device}' + \
           f'\n>>> CUDA: {"enabled" if Conf.UseCuda else "disabled"}' + \
           f'\n>>> GPU count: {Conf.GPUCount}' + \
           f'\n>>> Max. workers: {Conf.MaxWorkers}' + \
           f'\n>>> Data directory: {Conf.DataDir}' + \
           f'\n>>> Download data: {Conf.DownloadData}' + \
           f'\n>>> Data loader function: {Conf.DataLoader}' + \
           f'\n>>> Data loader arguments:'

    for key, val in Conf.DataLoadArgs.items():
        str += f'\n\t{key}: {val}'

    str += f'\n>>> Train batch size: {Conf.TrainBatchSize}' + \
           f'\n>>> Test batch size: {Conf.TestBatchSize}' + \
           f'\n>>> Train portion: {Conf.TrainPortion}' + \
           f'\n>>> Log directory: {Conf.LogDir}' + \
           f'\n>>> Log interval: {Conf.LogInterval}' + \
           f'\n>>> Output function: {Conf.OutputFunction.__name__ if Conf.OutputFunction is not None else None}' + \
           f'\n>>> Output function arguments:'

    for key, val in Conf.OutputFunctionArgs.items():
        str += f'\n\t{key}: {val}'

    str += f'\n>>> Loss function: {Conf.LossFunction.__name__ if Conf.LossFunction is not None else None}' + \
           f'\n>>> Optimiser: {Conf.Optimiser.__name__ if Conf.Optimiser is not None else None}' + \
           f'\n>>> Optimiser arguments:'

    for key, val in Conf.OptimiserArgs.items():
       str += f'\n\t{key}: {val}'

    str += f'\n>>> Unit test mode: {Conf.UnitTestMode}' + \
           f'\n=====================[ End of PyCortex configuration ]===================='

    print(str, file = _file)

def pause():

    key = ''
    while not (key == 'Y' or key == 'y'):
        key = input("Continue (Y/n)? ")
        if len(key) == 0:
            key = 'Y'

    return key

def optimise(_net,
             _data,
             _target,
             _optimiser,
             _loss_function,
             _output_function,
             _output_function_args = {}):

    def closure():

        _optimiser.zero_grad()
        output = _output_function(_net(_data), **_output_function_args)

#        print(f'Output: {output}')
#        print(f'Target: {_target}')
        loss = _loss_function(output, _target)
#        print(f'Loss: {loss}')
        loss.backward()
        _net.fitness.loss_stat.update(loss.item())
        return loss

    _net.optimise(closure, _optimiser)

def test(_net, _conf):

    _net.eval()
    test_loss = 0
    correct = 0

    loader = _conf.data_loader(_dir = _conf.data_dir,
                               _batch_size = _conf.test_batch_size,
                               _train = False,
                               _download = _conf.download_data,
                               **_conf.data_load_args)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(_conf.device), target.to(_conf.device)
            output = _conf.output_function(_net(data), **_conf.output_function_args)
            test_loss += _conf.loss_function(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)

    accuracy = 100. * correct / len(loader.dataset)
    print(f'[Net {_net.ID}] Test | Run {_conf.run} | ' +
          f'Epoch {_conf.epoch} Average loss: {test_loss:.4f}, ' +
          f'Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)')

    return accuracy

def train(_net, _conf):

    net = _net.to(_conf.device)

    # Train the network if it is not a new offspring
    if net.age > 0:
        net.train()
        if 'lr' not in _conf.optimiser_args:
            _conf.optimiser_args['lr'] = 1.0

        _conf.optimiser_args['lr'] *= (1.0 - net.fitness.relative) / (net.age + 1)

        optimiser = _conf.optimiser(net.parameters(), **_conf.optimiser_args)

        loader = _conf.data_loader(_dir = _conf.data_dir,
                                   _batch_size = _conf.train_batch_size,
                                   _train = True,
                                   _portion = net.complexity if _conf.train_portion is None else _conf.train_portion,
                                   **_conf.data_load_args)

        net.fitness.loss_stat.reset()

        examples = 0
        for batch_idx, (data, target) in enumerate(loader):

            examples += len(data)
            data, target = data.to(_conf.device), target.to(_conf.device)
            optimise(net, data, target, optimiser, _conf.loss_function, _conf.output_function, _conf.output_function_args)

        print(f'[Net {net.ID}] Train | Run {_conf.run} | Epoch {_conf.epoch} Trained on {100. * examples / len(loader.dataset):.2f}% of the dataset')

    # Evaluate on the test set to determine the fitness
    net.fitness.set(test(net, _conf))

    return net

def init():

    if get_rank() != 0:
        return

    """
    Initialise the ecosystem and generate initial species if speciation is enabled.
    """

    print("\n======[ Initialising ecosystem ]======\n")

    # Reset the static network attributes
    cn.Net.reset()

    # Reset the static species attributes
    cs.Species.reset()

    # Reset all statistics
    for key, val in Conf.Stats.items():
        Conf.Stats[key] = Stat.SMAStat()

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

            # Generate proto-species.
            proto_species = cs.Species(_genome = proto_net.get_genome())

            # Generate proto-nets for this proto-species.
            for n in range(net_quota):
                proto_net = cn.Net(_species = proto_species)

            if len(cs.Species.Populations) == cs.Species.Init.Count:
                break

            proto_net = cn.Net(_species = proto_species, _isolated = True)

            while cs.Species.find(proto_net.get_genome()) != 0:
                proto_net.mutate(_probabilities = probabilities, _structure = True, _parameters = True)

    else:

        # Generate proto-species.
        proto_species = cs.Species(_genome = proto_net.get_genome())

        # Generate proto-nets.
        for n in range(net_quota):
            proto_net = cn.Net(_species = proto_species)
            proto_net.mutate(_probabilities = probabilities, _structure = True, _parameters = True)

    print(f'Network count: {len(cn.Net.Ecosystem)}')
    for net in cn.Net.Ecosystem.values():
        print(net.as_str())

    print(f'Species count: {len(cs.Species.Populations)}')
    for species in cs.Species.Populations.values():
        print(species.as_str())

def calibrate(_run,
              _epoch):

    if get_rank() != 0:
        return

    print("\n======[ Calibrating ecosystem ]======\n")

    # Remove extinct species.
    extinct = [species.ID for species in cs.Species.Populations.values() if len(species.nets) == 0]

    while len(extinct) > 0:
        del cs.Species.Populations[extinct.pop()]

    # Reset the global champion.
    cn.Net.Champion = None

    # Running statistics about species fitness
    species_stat = Stat.SMAStat()

    # Highest fitness seen so far.
    top_fitness = -math.inf

    params = {}
    complexity_stat = Stat.SMAStat()
    for net in cn.Net.Ecosystem.values():
        params[net.ID] = net.get_parameter_count()
        complexity_stat.update(params[net.ID])

    for net_id in params.keys():
        # This means that for identical absolute fitness, an individual with
        # lower complexity would have a higher relative fitness.
        cn.Net.Ecosystem[net_id].complexity = complexity_stat.get_inv_offset(params[net_id])

    for species in cs.Species.Populations.values():

        # Compute the relative network fitness
        # and the absolute species fitness.
        species.calibrate()
        species_stat.update(species.fitness.absolute)

        if (cn.Net.Champion is None or
            cn.Net.Ecosystem[species.champion].fitness.absolute > top_fitness):

            cn.Net.Champion = species.champion
            top_fitness = cn.Net.Ecosystem[species.champion].fitness.absolute

    for species in cs.Species.Populations.values():
        # Compute the relative species fitness
        species.fitness.relative = species_stat.get_offset(species.fitness.absolute)

    print(f'>>> Global champion: {cn.Net.Champion} (fitness: {cn.Net.Ecosystem[cn.Net.Champion].fitness.absolute})')

    # Save the champion
    if cn.Net.Champion is not None:
        save_net(cn.Net.Ecosystem[cn.Net.Champion], f'/run_{_run}/epoch_{_epoch}', 'Champion')

    # Store statistics
    Conf.Stats['Species_count'].update(len(cs.Species.Populations))
    Conf.Stats['Network_count'].update(len(cn.Net.Ecosystem))
    Conf.Stats['Champion_parameter_count'].update(cn.Net.Ecosystem[cn.Net.Champion].get_parameter_count())
    Conf.Stats['Highest_fitness'].update(cn.Net.Ecosystem[cn.Net.Champion].fitness.absolute)

    epoch_stats = {
                  'Average_fitness': Stat.SMAStat(),
                  'Average_initial_fitness': Stat.SMAStat(),
                  'Champion_parameter_count': Stat.SMAStat()
                  }

    if not Conf.UnitTestMode:

        for net in cn.Net.Ecosystem.values():

            if net.ID == cn.Net.Champion:
                epoch_stats['Champion_parameter_count'].update(net.get_parameter_count())

            epoch_stats['Average_fitness'].update(net.fitness.absolute)

            if net.age == 0:
                epoch_stats['Average_initial_fitness'].update(net.fitness.absolute)

        # Global statistics
        for key, stat in Conf.Stats.items():
            Conf.Logger.add_scalar(key, stat.current_value, _epoch)

        # Epoch statistics
        for key, stat in epoch_stats.items():
            Conf.Logger.add_scalar(key, stat.mean, _epoch)
            stat.title = key
            save_stat(stat, f'/run_{_run}/epoch_{_epoch}')

def evolve(_run,
           _epoch):

    if get_rank() != 0:
       return

    print("\n======[ Evolving ecosystem ]======\n")

    # Set the offspring count to 0
    cs.Species.Offspring = 0

    if not Conf.UnitTestMode:

        # Increase the age of all networks.
        for net in cn.Net.Ecosystem.values():
            net.age += 1

        if _epoch < Conf.Epochs:

            # Evolve networks in all species.
            wheel = Rand.RouletteWheel()

            for species_id, species in cs.Species.Populations.items():
                wheel.add(species_id, species.fitness.relative)

            while not wheel.is_empty():
                cs.Species.Populations[wheel.pop()].evolve()

def cull():

    if get_rank() != 0:
        return

    print("\n======[ Culling ecosystem ]======\n")

    wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)

    for net_id, net in cn.Net.Ecosystem.items():
        if (net.age > 0 and
            net_id != cn.Net.Champion):
            # Old, unfit and simple networks
            # are more likely to be eliminated.
            wheel.add(net_id, net.fitness.relative * net.complexity / net.age)

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

    if len(removed_nets) > 0:
        print(f'Removed nets: {removed_nets}')
    if len(removed_species) > 0:
        print(f'Removed species: {removed_species}')

def execute():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    if Conf.UseCuda and rank > Conf.GPUCount:
        return

    if rank == 0:

        # Master process

        try:
            # Set the default tensor type
            if Conf.UseCuda:
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
            else:
                torch.set_default_tensor_type(torch.FloatTensor)

        except:
            print('Caught exception in init()')
            Conf.Tag = Tags.Exit
            dump_exception()

        if Conf.Evaluator is None:
            Conf.Evaluator = train

        assert Conf.DataLoader is not None, "Please assign a data loader function."

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

            print(f'\n===============[ Run {run} ]===============')

            # Initialise the ecosystem
            try:

                print("\n======[ Initialising ecosystem ]======\n")
                init()

                if not Conf.UnitTestMode:
                    os.makedirs(Conf.LogDir, exist_ok = True)
                    Conf.Logger = SummaryWriter(Conf.LogDir + f'/run_{run}')

            except:
                print('Caught exception in init()')
                Conf.Tag = Tags.Exit
                dump_exception()
                break

            for epoch in range(1, Conf.Epochs + 1):

                if len(Conf.Workers) == 0:
                    break

                print(f'\n===============[ Epoch {epoch} ]===============')

                print("\n======[ Evaluating ecosystem ]======\n")

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
                        package = (cn.Net.Ecosystem[net_id], Conf(run, epoch, worker if Conf.UseCuda else None))
                        comm.send(package, dest = worker, tag=Conf.Tag)

                # Wait for the last workers to finish
                while len(free_workers) < len(Conf.Workers):
                    manage_workers(free_workers)

                if len(Conf.Workers) > 0:
                    try:

                        # Compute the relative fitness of networks and species.
                        calibrate(run, epoch)

                        if epoch < Conf.Epochs:

                            # Evolve the ecosystem
                            evolve(run, epoch)

                            # Eliminate unfit networks and empty species.
                            if len(cn.Net.Ecosystem) > cn.Net.Max.Count:
                                cull()

                    except:
                        print('Caught exception in evolve()')
                        Conf.Tag = Tags.Exit
                        dump_exception()
                        break

            if os.path.exists(Conf.LogDir):
                with open(Conf.LogDir + '/config.txt', 'w+') as cfg_file:
                    print_conf(_file = cfg_file)

                # Save global statistics
                for key, stat in Conf.Stats.items():
                    stat.title = key
                    print(stat.as_str())
                    save_stat(stat)

        Conf.Tag = Tags.Exit

        print('\n\n>>> Sending exit command to workers...')
        for worker in Conf.Workers:
            comm.send(None, dest = worker, tag=Conf.Tag)

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

                    conf.evaluator(net, conf)
                    comm.send(net, dest=0, tag = Tags.Done)

                except Exception:
                    print(f'Caught exception in worker {rank} while evaluating network {net.ID}')
                    print(net.as_str(_parameters = True))
                    dump_exception()
                    break

            elif tag == Tags.Exit:
                break

        comm.send(None, dest=0, tag=Tags.Exit)

    print(f'Worker {rank} exiting...')
