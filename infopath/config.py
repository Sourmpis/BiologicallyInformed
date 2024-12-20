import numpy as np
import torch
import time
from types import SimpleNamespace
import os
import json
import pickle
from optparse import OptionParser
import copy


def compare_opt(log_path1, log_path2):
    opt1 = load_training_opt(log_path1)
    if log_path2 == "pseudodata":
        opt2 = config_pseudodata()
    else:
        opt2 = load_training_opt(log_path2)
    for key in vars(opt1).keys():
        if key not in vars(opt2).keys():
            # print(key, "this is doesn't exist")
            continue
        if key == "log_path":
            continue
        if vars(opt1)[key] != vars(opt2)[key]:
            print(key, vars(opt1)[key], vars(opt2)[key])


def load_training_opt(log_path):
    default = get_default_opt()
    opt = json.load(open(os.path.join(log_path, "opt.json"), "rb"))
    opt = SimpleNamespace(**opt)
    scale_function(opt)
    opt.device = torch.device(opt.device)
    opt_new = vars(default)
    opt = vars(opt)
    for i in opt_new:
        if i not in opt.keys():
            opt[i] = opt_new[i]
    opt = SimpleNamespace(**opt)
    return opt


def save_opt(log_path, opt):
    result_path_json = os.path.join(log_path, "opt.json")
    with open(result_path_json, "w") as f:
        optsave = copy.deepcopy(opt)
        optsave = vars(optsave)
        optsave["device"] = str(optsave["device"])
        optsave["scale_fun"] = str(optsave["scale_fun"]).split(" ")[1]
        json.dump(optsave, f)


def scale_function(opt):
    if opt.scale_fun == "sigmoid":
        fun = sigmoid
    elif opt.scale_fun == "linear":
        fun = linear
    elif opt.scale_fun == "log":
        fun = log
    else:
        raise NotImplementedError(
            "Scale Function: {} not implemented".format(opt.scale_fun)
        )
    opt.scale_fun = fun


def sigmoid(x, opt):
    if x != 0:
        return np.exp(x - 1) / (1 + np.exp(x - 1))
    else:
        return 0


def linear(x):
    return x


def log(x):
    return np.log(1 + x)


def get_default_opt():
    input_parameters = {
        # duration of stimulus in seconds
        "stim_duration": 0.01,
        # Frequency in Hz of the input neurons
        "input_f0": 2.0,
        # Number of inputs to the RNNs
        "n_rnn_in": 128,
        # function to scale differnt stim amplitudes, it can be {'sigmoid', 'linear', 'log'}
        "scale_fun": "log",
        # valance of the stimulus, multiplier of scale_fun
        "stim_valance": 11.5,
        # delay from input to rnn in seconds
        "thalamic_delay": 0.005,
    }
    optimizer_parameters = {
        # learning rate
        "lr": 0.0003,
        # weight decay
        "w_decay": 0.01,
        # l1 decay
        "l1_decay": 0.01,
        # maximum iteration before stopping the early stopping
        "early_stop": 2000,
        # wheter to penalize across area connections
        "l1_decay_across": 0.001,
        # a different way to implement sparsity (this might be better in the future)
        "iterative_pruning": False,
    }
    network_parameters = {
        #  how many neurons
        "n_units": 300,
        # how strong is the membrane potential noise, in each area
        "noise_level_list": [0.16, 0.16, 0.16],
        # timeconstant of membrane potential of areas in ms
        "tau_list": [10.0, 10.0, 10.0],
        # timeconstant of adaptative threshold in ms
        "tau_adaptation": 144.0,
        # name of areas
        "areas": ["wS1", "mPFC", "tjM1"],
        # smallest synaptic delay
        "n_delay": 5,
        # longest synaptic delay
        "inter_delay": 5,
        # ratio of exc to inh membrane timeconstant
        "exc_inh_tau_mem_ratio": 2.0,
        # start and endpoint for reaction_time limits defaults none values
        "reaction_time_limits": None,
        # which areas project to motor decoder
        "motor_areas": [],
        # maximum delay for neural activity to generate jaw/tongue
        "jaw_delay": 40,
        # min delay for neural activity to generate jaw/tongue
        "jaw_min_delay": 12,
        # timeconstant of jaw/tongue integration in ms
        "tau_jaw": 100,
        # restrict inter inh->exc connections
        "restrict_inter_area_inh": 1,
        # propability a neuron to be adaptive
        "prop_adaptive": 0.0,
        # spike function type {"bernoulli", "deterministic"}
        "spike_function": "bernoulli",
        # train bias (offset in the v_{rest})
        "train_bias": False,
        # train bias(a multiplicative factor in the membrane noise)
        "train_noise_bias": False,
        # the bias is applied in threshold or the membrane
        "bias_in_mem": True,
        # if true there is a membrane offset per neuron
        "trial_offset": False,
        # number of latent spaces of trial noise
        "latent_space": 2,
        # to train the tau_{adaptation} or not
        "train_adaptation": False,
        "train_delays": False,
        # proportion of excitatory neurons
        "p_exc": 0.85,
        # temperature for sigmoid in bernoulli spike function
        "temperature": 7.5,
        # if to use the MLP trial offset (this the one described in the paper)
        "latent_new": False,
        # if the neuron model is LIF or conductance based
        "conductance_based": False,
        # if jaw/tongue feeds back to the RSNN
        "jaw_open_loop": False,
        # if to use jaw or tongue
        "jaw_tongue": 1,
        # if to use a nonlinear transformation for jaw/tongue
        "jaw_nonlinear": False,
        # if to scale the jaw/tongue in the model
        "scaling_jaw_in_model": False,
        # percentage of exc neurons in input neurons
        "p_exc_in": 0.8,
        # what is the initial v_{rest} values
        "v_rest": 0,
        # what is the initial threshold values
        "thr": 0.1,
        # don't allow certain areas to connect to other areas
        "block_graph": [],
        # how many populations per area to use in PopRSNN
        "pop_per_area": 1,
        # whehter to have exc-inh connections
        "flag_ei": True,
        # whether to have connections based on distance between areas
        "weights_distance_based": False,
        # whether to have connections based on random distances between areas
        "weights_random_delays": False,
        # whether to train the input connections
        "train_input_weights": True,
        # pee == pei == pie == pii
        "p": 1,
        # if to bound the trial offset
        "trial_offset_bound": False,
        # whether to reset membrane potential after each spike
        "with_reset": True,
    }
    training_parameters = {
        # how often to log
        "log_every_n_steps": 100,
        # batch size
        "batch_size": 50,
        # Number of training steps
        "n_steps": 20000,
        # how impact has the main loss to the total loss
        "coeff_loss": 1.0,
        # loss over single neuron or first averaging over population
        "loss_neuron_wise": 0,
        # loss over single trial
        "loss_trial_wise": 0,
        "coeff_trial_loss": 1,
        # loss for baseline firing rate
        "loss_firing_rate": 1,
        "coeff_fr_loss": 0,
        # if the T_trial is area specific
        "trial_loss_area_specific": True,
        # if the T_trial is neurotransmitter specific
        "trial_loss_exc_specific": False,
        # if to use sinkhorn loss or hungarian algorithm
        "geometric_loss": False,
        # if to z-score T_neuron and T_trial
        "z_score": True,
        # if to use task-splitter
        "with_task_splitter": False,
        # if to use T_trial with a GAN, what we call spikeT-GAN
        "t_trial_gan": False,
        # if to use the trial-matched MLE loss
        "loss_trial_matched_mle": False,
        # if to penalize large voltage traces
        "loss_mem_volt": 0,
        # whether to use the cross-correlation loss version 0 or 1
        "cc_version": 0,
        # this refers to the appendix of my thesis where we have a correction term for the cross-correlation loss
        "stats_loss": False,
        # for the teacher training
        "diff_strength": 1.0,
        # starting to remove the average and use svd features (not yet operational)
        "feat_svd": False,
    }
    general_parameters = {
        # directory of the dataset
        "datapath": "./datasets",
        # Device to use {either cpu or cuda:0}
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        # gauss std for filtering spikes beforer loss calculation in ms
        "psth_filter": 5,
        # second filtering to build the T_trial, stride length in ms
        "pop_avg_resample": 2,
        # if verbose print all messages else no
        "verbose": 1,
        # percentage of hidden neurons(neurons that do not correspond to recordings)
        "hidden_perc": 0.0,
        # simplified runs rsnn.py, mean runs pop_mean.py
        "lsnn_version": "simplified",
        # seed the training
        "seed": 0,
        # clip the grad (avoid exploding gradients)
        "clip_grad": 0,
    }
    data_parameters = {
        # start in seconds relative to onset
        "start": -0.1,
        # stop in seconds relative to onset
        "stop": 0.5,
        # Onset of trial in seconds
        "trial_onset": 1.0,
        # what are the stimuli onsets in seconds
        "stim_onsets": [0, 1],
        # Timestep of simulation in ms
        "dt": 1.0,
        # more readable way to set stim
        "stim": [1, 2, 3, 4],
        # which trial types to load from the data
        "trial_types": [0, 1, 2, 3],
        # if to load the data with behaviour
        "with_behaviour": False,
    }
    default = {}
    default.update(input_parameters)
    default.update(optimizer_parameters)
    default.update(network_parameters)
    default.update(training_parameters)
    default.update(general_parameters)
    default.update(data_parameters)

    default["time"] = time.ctime()
    default["num_areas"] = len(default["areas"])
    default["device"] = torch.device(default["device"])
    default = SimpleNamespace(**default)
    scale_function(default)
    return default


# # PseudoData
def config_pseudodata():
    opt = get_default_opt()
    opt.datapath = "./datasets/PseudoData_3areas_abstract"
    opt.areas = ["area1", "area2", "area3"]
    opt.n_units = 750
    opt.block_graph = []
    # opt.trial_types = [0, 1]
    opt.trial_types = [i for i in range(4)]
    # opt.datapath = "./datasets/PseudoData_3areas_chain"
    # opt.datapath = "./datasets/PseudoData_3areas_parallel"
    # opt.areas = ["area1", "area2", "area3"]
    # opt.n_units = 750
    # opt.block_graph = []
    # # chain / mechanism 1
    # opt.block_graph = [[1, 0], [2, 0], [2, 1], [0, 2]]
    # parallel / mechanism 2
    # opt.block_graph = [[1, 0], [2, 0], [2, 1], [1, 2]]
    # # mechanism 3
    # opt.block_graph = [[2, 0], [2, 1], [0, 2], [0, 1]]

    # # 2 areas / mechanism 1
    # opt.block_graph = [[1, 0]]
    # # 2 areas / mechanism 2
    # opt.block_graph = [[0, 1]]

    opt.trial_onset = 1
    opt.hidden_perc = 0.0

    opt.stim = [4]
    opt.num_areas = len(opt.areas)
    opt.n_rnn_in = 1
    opt.p_exc_in = 1
    opt.start, opt.stop = -0.1, 0.3
    opt.batch_size = 300
    opt.train_bias = False
    opt.train_noise_bias = False
    opt.train_adaptation = False
    opt.prop_adaptive = 0.0
    opt.noise_level_list = [0.1 for i in range(len(opt.areas))]
    opt.input_f0 = 5
    opt.lsnn_version = "simplified"  # "mean"
    opt.thalamic_delay = 0.004  # * (opt.lsnn_version != "srm")
    opt.tau_list = [10 for i in range(opt.num_areas)]
    opt.exc_inh_tau_mem_ratio = 3.0
    opt.stim_onsets = [0]

    opt.restrict_inter_area_inh = True
    opt.dt = 4
    opt.n_delay = 4
    opt.inter_delay = 4
    opt.rec_groups = 1
    opt.psth_filter = 8  # miliseconds # int(psth_filter / opt.dt)

    opt.trial_offset = True
    opt.latent_space = 5
    opt.latent_new = False
    opt.gan_loss = False
    opt.t_trial_gan = False
    opt.loss_trial_matched_mle = 0
    opt.loss_trial_wise = 1
    opt.loss_neuron_wise = 1
    opt.loss_cross_corr = 0
    opt.coeff_loss = 1
    opt.coeff_fr_loss = 1
    opt.coeff_trial_loss = 1
    opt.coeff_cross_corr_loss = 5000
    opt.early_stop = 8000
    opt.coeff_trial_fr_loss = 0.0
    opt.loss_firing_rate = 1

    opt.w_decay = 0.0
    opt.l1_decay = 0.0

    opt.lr = 0.001
    opt.p_exc = 0.8
    opt.trial_loss_area_specific = True
    opt.geometric_loss = False
    opt.motor_areas = []
    opt.jaw_delay = 40
    opt.tau_jaw = 50
    opt.jaw_version = 1

    opt.conductance_based = False
    opt.gan_hidden_neurons = 128
    opt.spike_function = "bernoulli"

    opt.with_behaviour = False
    opt.loss_trial_type = False
    opt.with_task_splitter = False
    opt.temperature = 5
    opt.v_rest = 0  # -75  #
    opt.thr = 0.1  # -50  #
    opt.trial_offset_bound = False
    opt.trial_mle_with_vae = False
    opt.num_areas = len(opt.areas)
    opt.bottleneck_neurons = 30
    opt.clip_grad = 500
    # opt.device = "cpu"
    return opt


# # Vahid
def config_vahid():
    opt = get_default_opt()
    opt.datapath = "./datasets/DataFromVahid_expert"
    opt.areas = ["wS1", "wS2", "wM1", "wM2", "ALM", "tjM1"]
    opt.num_areas = len(opt.areas)
    opt.stim = [0, 1]
    opt.n_units = 750
    opt.n_rnn_in = 2
    opt.start, opt.stop = -0.2, 1.2
    opt.noise_level_list = [0.10 for i in range(len(opt.areas))]

    opt.prop_adaptive = 0.0
    opt.input_f0 = 5
    opt.tau_list = [10 for i in range(opt.num_areas)]

    opt.dt = 4
    opt.inter_delay = 4  # miliseconds
    opt.n_delay = 4
    opt.rec_groups = 1

    opt.psth_filter = 8  # miliseconds # int(psth_filter / opt.dt)
    opt.lsnn_version = "simplified"
    opt.thalamic_delay = 0.004  # * (opt.lsnn_version != "srm")
    opt.early_stop = 8000
    opt.lr = 1e-3
    opt.p_exc = 0.8
    opt.p_exc_in = 1
    opt.batch_size = 150

    opt.loss_neuron_wise = 1
    opt.loss_cross_corr = 0
    opt.loss_trial_wise = 1
    opt.loss_firing_rate = 0
    opt.coeff_loss = 1
    opt.coeff_fr_loss = 1
    opt.coeff_trial_fr_loss = 0.0000
    opt.coeff_trial_loss = 1
    opt.coeff_cross_corr_loss = 5000

    opt.trial_offset = False
    opt.latent_space = 5
    opt.train_noise_bias = False
    opt.train_adaptation = False
    opt.train_bias = False
    opt.spike_function = "bernoulli"
    opt.w_decay = 0.0
    opt.l1_decay = 0.0

    opt.exc_inh_tau_mem_ratio = 3.0
    opt.conductance_based = False
    opt.trial_loss_area_specific = True
    opt.geometric_loss = True

    opt.motor_areas = [4, 5]
    opt.jaw_delay = 40
    opt.jaw_min_delay = 12
    opt.tau_jaw = 50
    opt.jaw_version = 1

    opt.gan_loss = False
    opt.gan_hidden_neurons = 128
    opt.latent_new = False
    opt.with_behaviour = True
    # opt.device = "cpu"
    opt.reaction_time_limits = [-1, 0.3]
    opt.with_task_splitter = True
    opt.z_score = True
    opt.jaw_open_loop = True
    opt.scaling_jaw_in_model = True
    opt.jaw_tongue = 1
    opt.jaw_nonlinear = False
    opt.temperature = 1
    opt.v_rest = 0  # -75  #
    opt.thr = 0.1  # -50  #
    opt.trial_offset_bound = False
    opt.num_areas = len(opt.areas)
    opt.clip_grad = 2000
    return opt


def get_opt(log_path=None):
    default = get_default_opt()
    if log_path == None:
        return copy.copy(default)
    else:
        opt = load_training_opt(log_path)
        for key in vars(default).keys():
            if vars(opt).get(key) is None:
                setattr(opt, key, getattr(default, key))
        return opt


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--config", type="string", default="none")
    (pars, _) = parser.parse_args()

    config_path = os.path.join("configs", pars.config)
    if ~os.path.exists(config_path):
        os.mkdir(config_path)

    default = get_default_opt()
    opt = copy.copy(default)
    # opt = config_vahid()
    opt = config_pseudodata()
    save_opt(config_path, opt)
