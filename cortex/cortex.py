import argparse
import os
import sys
import math
from datetime import datetime

import torch
torch.set_printoptions(precision = 4, threshold = 5000, edgeitems = 5, linewidth = 160)

import torch.nn.functional as tnf
import torch.multiprocessing as tm
tm.set_sharing_strategy('file_system')

from tensorboardX import SummaryWriter

import cortex.random as Rand
import cortex.statistics as Stat
import cortex.functions as Func

import cortex.network as cn
import cortex.layer as cl
import cortex.species as cs

################[ Global variables ]################

class Conf:
    ExperimentName = 'Experiment'
    Runs = 1
    Epochs = 50

    TrainBatchSize = 64
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

def init_conf():

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

    print("\n========================[ PyCortex configuration ]========================\n",
          "\nExperiment name:", Conf.ExperimentName,
          "\nRuns:", Conf.Runs,
          "\nEpochs:", Conf.Epochs,
          "\nInit. networks:", cn.Net.Init.Count,
          "\nMax. networks:", cn.Net.Max.Count,
          "\nSpeciation:", 'enabled' if cs.Species.Enabled else 'disabled',
          file = _file)

    if cs.Species.Enabled:
        print("\tInit. species:", cs.Species.Init.Count,
              "\n\tMax. species:", cs.Species.Max.Count,
              file = _file)

    print("\nLearning rate:", Conf.LearningRate,
          "\nMomentum:", Conf.Momentum,
          "\nCUDA:", 'enabled' if Conf.UseCuda else 'disabled',
          "\nInput shape:", cn.Net.Input.Shape,
          "\nOutput shape:", cn.Net.Output.Shape,
          "\nLayer bias:", cl.Layer.Bias,
          "\nLayer activations:",
          "\nLayers:",
          file = _file)

    for key, val in cl.Layer.Activations.items():
        print('\t', key, ':', val.__name__)

    for layer_index, layer_def in enumerate(cn.Net.Init.Layers):
        print("\tLayer %r:" % layer_index, file = _file)
        layer_def.print(_file = _file)

    print("\nInit. function:", cl.Layer.InitFunction.__name__,
          "\nInit. arguments:",
          file = _file)

    for key, val in cl.Layer.InitArgs.items():
        print("\t", key, ":", val, file = _file)

    print("\nMax. nets:", cn.Net.Max.Count,
          "\nMax. net age:", cn.Net.Max.Age,
          "\nSpeciation:", "enabled" if cs.Species.Enabled else "disabled",
          file = _file)

    if cs.Species.Enabled:
        print("Init. species:", cs.Species.Init.Count,
              "\nMax. species:", cs.Species.Max.Count,
              file = _file)

    print("\nLearning rate:", Conf.LearningRate,
          "\nMomentum:", Conf.Momentum,
          "\nDevice:", Conf.Device,
          "\nMax. workers:", Conf.MaxWorkers,
          file = _file)

    print("\nData loader arguments:\n", file = _file)
    for key, val in Conf.DataLoadArgs.items():
        print("\t", key, ":", val, file = _file)

    print("Data directory:", Conf.DataDir,
          "\nDownload:", Conf.DownloadData,
          file = _file)

    print("Optimiser:", Conf.Optimiser.__name__,
          "\nLoss function:", Conf.LossFunction.__name__,
          "\nLog directory:", Conf.LogDir,
          "\nLog interval:", Conf.LogInterval,
          "\nUnit test mode:", Conf.UnitTestMode,
          file = _file)

    print("\n=====================[ End of PyCortex configuration ]====================\n", file = _file)

def pause():
    key = input("Continue (Y/n)? ")
    if len(key) == 0:
        key = 'Y'
    return key

def init():
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
                'stride': 1
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
                proto_net.mutate(_parameters = False, _probabilities = probabilities)

    else:

        # Generate proto-species
        proto_species = cs.Species(_genome = proto_net.get_genome())

        # Generate proto-nets.
        for n in range(net_quota):
            proto_net = cn.Net(_species = proto_species)
            proto_net.mutate(_parameters = False)

    print("Nets:", len(cn.Net.Ecosystem))
    for net in cn.Net.Ecosystem.values():
        net.print()

    print("Species:", len(cs.Species.Populations))
    for species in cs.Species.Populations.values():
        species.print()

    if not Conf.UnitTestMode:
        os.makedirs(Conf.LogDir, exist_ok = True)
        Conf.Logger = SummaryWriter(Conf.LogDir + '/TensorBoard')

def calibrate():

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
        print("Network", net_id, "fitness scale:", complexity_fitness_scale[net_id])

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
        species.print()

def save(_net_id,
         _run,
         _epoch,
         _name = None):

    save_dir = Conf.LogDir + '/run_' + str(_run) + '/epoch_' + str(_epoch)

    os.makedirs(save_dir, exist_ok = True)

    if _name is None:
        name = 'net_' + str(_net_id)
    else:
        name = _name

    torch.save(cn.Net.Ecosystem[_net_id], save_dir + '/' + name + '.pt')

    with open(save_dir + '/' + name + '.txt', 'w+') as plaintext:
        cn.Net.Ecosystem[_net_id].print(_file = plaintext)

def cull():

    champions = [species.champion for species in cs.Species.Populations.values()]

    species_wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)

    # Networks from species that haven't made much progress for a while
    # are more likely to be selected for culling.
    for species_id, species in cs.Species.Populations.items():
        species_wheel.add(species_id, species.fitness.relative * species.fitness.stat.get_sd())

    while len(cn.Net.Ecosystem) > cn.Net.Max.Count:

#        # Get a random species ID
#        species_wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)
#        for species in cs.Species.Populations.values():
#            species_wheel.add(species.ID, species.fitness.relative)

        species_id = species_wheel.spin()

        # Get a random network ID

        wheel = Rand.RouletteWheel(Rand.WeightType.Inverse)
        for net_id in cs.Species.Populations[species_id].nets:
            net = cn.Net.Ecosystem[net_id]
            if (net.age > 0 and
                net_id not in champions):
                # Networks whose fitness is low or hasn't changed much for a while
                # are more likely to be eliminated.
                wheel.add(net_id, net.fitness.relative * net.fitness.stat.get_sd() / net.age)

        net_id = wheel.spin()

        if net_id is not None:

            print('Removing network {} from species {}'.format(net_id, species_id))

            # Remove the network from the species.
            cs.Species.Populations[species_id].nets.remove(net_id)

            # Remove the network from the ecosystem.
            del cn.Net.Ecosystem[net_id]

        if len(cs.Species.Populations[species_id].nets) == 0:
            del cs.Species.Populations[species_id]

def evolve(_stats,
           _run,
           _epoch):

    print("======[ Evolving ecosystem ]======")

    # Compute the relative fitness of networks and species.
    print("\t`-> Calibrating...")
    calibrate()

    # Set the offspring count to 0
    cs.Species.Offspring = 0

    if not Conf.UnitTestMode:
        for net in cn.Net.Ecosystem.values():

#            Conf.Logger.add_scalars('Stats for network ' + str(net.ID), {
#                                    'Absolute fitness': net.fitness.absolute,
#                                    'Relative fitness': net.fitness.relative,
#                                    'Layers': len(net.layers),
#                                    'Parameters': net.get_parameter_count()
#                                    },
#                        _epoch)
            save(net.ID, _run, _epoch)

        Conf.Logger.add_scalar('Highest fitness', cn.Net.Ecosystem[cn.Net.Champion].fitness.absolute, _epoch)
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
            print("\t`-> Culling...")
            cull()

def run():

    assert Conf.Evaluator is not None, "Please assign a function for training networks."

    stats = {
            'Parameters': Stat.SMAStat('Parameters'),
            'Accuracy': Stat.SMAStat('Accuracy')
            }

    context = tm.get_context('spawn')

    shared_conf = tm.Manager().Namespace()

    shared_conf.data_dir = Conf.DataDir
    shared_conf.data_load_args = Conf.DataLoadArgs
    shared_conf.train_batch_size = Conf.TrainBatchSize
    shared_conf.test_batch_size = Conf.TestBatchSize
    shared_conf.device = Conf.Device
    shared_conf.loss_function = Conf.LossFunction
    shared_conf.output_function = Conf.OutputFunction
    shared_conf.output_function_args = Conf.OutputFunctionArgs
    shared_conf.optimiser = Conf.Optimiser
    shared_conf.log_interval = Conf.LogInterval

    for run in range(1, Conf.Runs + 1):

        init()

        print("===============[ Run", run, "]===============")

        for epoch in range(1, Conf.Epochs + 1):

            print("======[ Epoch", epoch, "]======")

            print("\t`-> Evaluating networks...")

            with context.Pool(processes = Conf.MaxWorkers) as pool:
                nets = pool.starmap(Conf.Evaluator, zip(cn.Net.Ecosystem.values(), [epoch] * len(cn.Net.Ecosystem), [shared_conf] * len(cn.Net.Ecosystem)))
                pool.close()
                pool.join()

            for net in nets:
                cn.Net.Ecosystem[net.ID] = net

            evolve(stats, run, epoch)

        for net_id in cn.Net.Ecosystem.keys():
            save(net_id, run, epoch)

    with open(Conf.LogDir + '/config.txt', 'w+') as cfg_file:
        print_conf(_file = cfg_file)

    for key, stat in stats.items():
        with open(Conf.LogDir + '/' + str(key) + '_stat.txt', 'w+') as stat_file:
            stat.print(_file = stat_file)
