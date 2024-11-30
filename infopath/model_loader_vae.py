from infopath.session_stitching import build_network
from models.pop_rsnn import PopRSNN
from models.rsnn import RSNN
from datasets.prepare_input import InputSpikes
import numpy as np
import torch.nn as nn
import torch
from infopath.utils.logger import reload_weights
import os
import copy
from infopath.loss_splitter import NormalizedMultiTaskSplitter
from infopath.losses import *


class FullModel(nn.Module):
    def __init__(self, opt, rsnn, encoder):
        super(FullModel, self).__init__()
        self.opt = opt
        self.num_areas = opt.num_areas
        self.rsnn = rsnn
        self.encoder = encoder
        # output dim is the number of neurons in the network times (the number of delays + 1 for voltage + 1 for adaptation) + 1 for jaw
        output_dim = opt.n_units * 1 + opt.latent_space
        self.fc_bottleneck_to_state = nn.Linear(opt.bottleneck_neurons, output_dim)
        self.input_spikes = InputSpikes(opt, opt.input_seed)
        self.timestep = self.opt.dt * 0.001
        # some filter functions
        kernel_size1 = int(self.opt.spike_filter_std / self.opt.dt)
        padding = int((kernel_size1 - 1) / 2)
        self.filter1 = torch.nn.AvgPool1d(
            kernel_size1,
            int(kernel_size1 // 2),
            padding=padding,
            count_include_pad=False,
        )
        stride2 = opt.resample
        padding = int((stride2 * 2 - 1) / 2)
        self.filter2 = torch.nn.AvgPool1d(
            2 * stride2, stride2, padding=padding, count_include_pad=False
        )

        self.T = int((self.opt.stop - self.opt.start) / self.opt.dt * 1000)
        self.jaw_mean = torch.nn.Parameter(torch.zeros(1))
        self.jaw_std = torch.nn.Parameter(torch.ones(1) * 0.1)

    def filter_fun1(self, spikes):
        """filter spikes

        Args:
            spikes (torch.tensor): spikes with dimensionality time x trials x neurons
        Return:
            filtered spikes with kernel specified from the self.opt
        """
        if spikes is None:
            return None
        spikes = spikes.permute(2, 1, 0)
        spikes = self.filter1(spikes)
        return spikes.permute(2, 1, 0)

    def filter_fun2(self, x):
        """filter signal

        Args:
            x (torch.tensor): signal with dimensionality time x trials x neurons
        Return:
            filtered signals with kernel specified from the self.opt
        """
        if x is None:
            return None
        x = x.permute(2, 1, 0)
        x = self.filter2(x)
        return x.permute(2, 1, 0)

    @torch.no_grad()
    def steady_state(self, state=None, sample_trial_noise=True):
        """
        set state of network in the steady state (basically run the network for 300ms
        from a zero state) we don't train this part so torch.no_grad
        """
        trials = torch.zeros(self.opt.batch_size).long().to(self.opt.device)
        spike_data = self.input_spikes_pre(trials)
        if state is None:
            state = self.rsnn.zero_state(self.opt.batch_size)
        _, _, _, state = self.rsnn(
            spike_data, state, sample_trial_noise=sample_trial_noise
        )
        return state

    def step(self, input_spikes, state, mem_noise, start=None, stop=None):
        opt = self.opt
        if start is None:
            start = 0
        if stop is None:
            stop = input_spikes.shape[0]
        self.rsnn.mem_noise = mem_noise[start:stop]
        spike_outputs, voltages, model_jaw, state = self.rsnn(
            input_spikes[start:stop].to(opt.device),
            state,
            sample_trial_noise=False,
            sample_mem_noise=False,
        )
        if not opt.scaling_jaw_in_model:
            model_jaw = (model_jaw - self.jaw_mean) / self.jaw_std
            if self.opt.jaw_nonlinear:
                model_jaw = torch.exp(model_jaw) - 1
        return spike_outputs, voltages, model_jaw, state

    def from_bottleneck_to_state(self, bottleneck, batch_size):
        state = self.rsnn.zero_state(batch_size)
        v_state = bottleneck[:, : self.opt.n_units][None]
        self.rsnn.trial_noise = bottleneck[:, self.opt.n_units :]
        v_state = torch.tanh(v_state) * (self.opt.thr - self.opt.v_rest)
        state[1].data = v_state
        self.rsnn.sample_mem_noise(50, batch_size)
        mem_noise = self.rsnn.mem_noise.clone()
        inp_spikes = self.input_spikes(torch.zeros(batch_size).long())[:50]
        _, _, _, state = self.step(inp_spikes, state, mem_noise)
        return state

    def forward(self, stims, data_spikes, sess_id, with_encoder=True):
        self.rsnn.sample_trial_noise(stims.shape[0])
        if with_encoder:
            bottleneck = self.encoder(data_spikes, sess_id)
            mu = bottleneck[:, : self.opt.bottleneck_neurons]
            logvar = bottleneck[:, self.opt.bottleneck_neurons :]
            std = torch.exp(logvar / 2)
            bottleneck = mu + torch.randn(mu.shape).to(self.opt.device) * std
        else:
            bottleneck = torch.randn(stims.shape[0], self.opt.bottleneck_neurons)
            bottleneck = bottleneck.to(self.opt.device)
            mu = logvar = bottleneck
        state = self.fc_bottleneck_to_state(bottleneck)
        state = self.from_bottleneck_to_state(state, stims.shape[0])
        input_spikes = self.input_spikes(stims)
        self.rsnn.sample_mem_noise(self.T, stims.shape[0])
        mem_noise = self.rsnn.mem_noise.clone()
        return input_spikes, self.step(input_spikes, state, mem_noise), mu, logvar

    def step_with_dt(
        self, input_spikes, state, light=None, dt=25, sample_mem_noise=False
    ):  # not really used in the paper, usefull for running forward the network without training in smaller gpus
        if sample_mem_noise:
            self.rsnn.sample_mem_noise(input_spikes.shape[0], input_spikes.shape[1])
        mem_noise = self.rsnn.mem_noise.clone()
        spikes, voltages, jaw, l = [], [], [], None
        for i in range(np.ceil(input_spikes.shape[0] / dt).astype(int)):
            self.rsnn.mem_noise = mem_noise[i * dt : (i + 1) * dt].clone()
            if light is not None:
                l = light[i * dt : (i + 1) * dt]
            sp, v, j, state = self.rsnn(
                input_spikes[i * dt : (i + 1) * dt],
                state,
                l,
                sample_mem_noise=False,
                sample_trial_noise=False,
            )
            spikes.append(sp[0])
            voltages.append(v[0])
            jaw.append(j[0])
        spikes = torch.cat(spikes, dim=0)
        voltages = torch.cat(voltages, dim=0)
        jaw = torch.cat(jaw, dim=0)
        self.rsnn.mem_noise = mem_noise
        if not self.opt.scaling_jaw_in_model:
            jaw = (jaw - self.jaw_mean) / self.jaw_std
            if self.opt.jaw_nonlinear:
                model_jaw = torch.exp(model_jaw) - 1
        return spikes, voltages, jaw, state

    def mean_activity(self, activity, clusters=None):
        with torch.no_grad():
            device = self.opt.device
            activity = self.filter_fun1(activity.to(device)).cpu()
            if clusters is None:
                clusters = torch.arange(activity.shape[2]) > 0
            step = self.timestep
            activity = activity[..., clusters]
            exc_index = self.rsnn.excitatory_index[clusters]
            area = self.rsnn.area_index[clusters]
            mean_exc, mean_inh = [], []
            for i in range(self.num_areas):
                area_index = area == i
                exc_mask = exc_index & area_index
                exc_mask = exc_mask.cpu()
                simulation_exc = (
                    np.nanmean(activity[..., exc_mask].cpu(), (1, 2)) / step
                )
                mean_exc.append(simulation_exc)
                inh_index = ~exc_index
                inh_mask = inh_index & area_index
                inh_mask = inh_mask.cpu()
                simulation_inh = (
                    np.nanmean(activity[..., inh_mask].cpu(), (1, 2)) / step
                )
                mean_inh.append(simulation_inh)
        return mean_exc, mean_inh


def load_model_and_optimizer(opt, reload=False, last_best="last"):
    rsnn = init_rsnn(opt)
    if os.path.exists(os.path.join(opt.log_path, "sessions.npy")):
        session = np.load(os.path.join(opt.log_path, "sessions.npy"), allow_pickle=True)
        neuron_index = np.load(os.path.join(opt.log_path, "neuron_index.npy"))
        firing_rate = np.load(os.path.join(opt.log_path, "firing_rate.npy"))
        area = np.load(os.path.join(opt.log_path, "areas.npy"), allow_pickle=True)
    else:
        neuron_index, firing_rate, session, area = build_network(
            rsnn, opt.datapath, opt.areas, opt.with_behaviour, opt.hidden_perc
        )
    encoder = Encoder(
        int((opt.stop - opt.start) / opt.dt * 1000),
        [(session == sess).sum() for sess in np.unique(session)],
        opt.n_units,
        opt.bottleneck_neurons,
    )

    model = FullModel(opt, rsnn, encoder)
    model.neuron_index = neuron_index
    model.firing_rate = firing_rate
    model.areas = area
    model.sessions = session

    # params =  list(model.rsnn.parameters()) + list(
    #     model.fc_bottleneck_to_state.parameters()
    # )
    optimizerG = torch.optim.AdamW(
        model.parameters(), lr=opt.lr, weight_decay=opt.w_decay
    )

    if reload:
        reload_weights(opt, model, optimizerG, last_best=last_best)

    model.to(opt.device)
    return model, None, optimizerG, None


def init_rsnn(opt):
    if opt.lsnn_version != "mean":
        rsnn = RSNN(
            opt.n_rnn_in,
            opt.n_units,
            sigma_mem_noise=opt.noise_level_list,
            num_areas=opt.num_areas,
            tau_adaptation=opt.tau_adaptation,
            tau=opt.tau_list,
            exc_inh_tau_mem_ratio=opt.exc_inh_tau_mem_ratio,
            n_delay=opt.n_delay,
            inter_delay=opt.inter_delay,
            restrict_inter_area_inh=opt.restrict_inter_area_inh,
            prop_adaptive=opt.prop_adaptive,
            dt=opt.dt,
            p_exc=opt.p_exc,
            spike_function=opt.spike_function,
            train_v_rest=opt.train_bias,
            trial_offset=opt.trial_offset,
            rec_groups=opt.rec_groups,
            latent_space=opt.latent_space,
            train_adaptation=opt.train_adaptation,
            train_noise_bias=opt.train_noise_bias,
            conductance_based=opt.conductance_based,
            jaw_delay=opt.jaw_delay,
            jaw_min_delay=opt.jaw_min_delay,
            tau_jaw=opt.tau_jaw,
            motor_areas=opt.motor_areas,
            temperature=opt.temperature,
            latent_new=opt.latent_new,
            jaw_open_loop=opt.jaw_open_loop,
            scaling_jaw_in_model=opt.scaling_jaw_in_model,
            p_exc_in=opt.p_exc_in,
            v_rest=opt.v_rest,
            thr=opt.thr,
            trial_offset_bound=opt.trial_offset_bound,
        )
    else:
        rsnn = PopRSNN(
            opt.n_rnn_in,
            opt.n_units,
            sigma_mem_noise=opt.noise_level_list,
            num_areas=opt.num_areas,
            tau=opt.tau_list,
            n_delay=opt.n_delay,
            inter_delay=opt.inter_delay,
            dt=opt.dt,
            p_exc=opt.p_exc,
            rec_groups=opt.rec_groups,
            latent_space=opt.latent_space,
            train_noise_bias=opt.train_noise_bias,
            jaw_delay=opt.jaw_delay,
            jaw_min_delay=opt.jaw_min_delay,
            tau_jaw=opt.tau_jaw,
            motor_areas=opt.motor_areas,
            latent_new=opt.latent_new,
            jaw_open_loop=opt.jaw_open_loop,
            scaling_jaw_in_model=opt.scaling_jaw_in_model,
            p_exc_in=opt.p_exc_in,
            trial_offset=opt.trial_offset,
            pop_per_area=opt.pop_per_area,
        )
    return rsnn


class Encoder(nn.Module):
    def __init__(self, time, input_neurons, hidden_neurons, bottleneck_neurons):
        super().__init__()
        self.fc1 = nn.ModuleList(
            [nn.Linear(inp, hidden_neurons) for inp in input_neurons]
        )
        self.conv1d = nn.Conv1d(hidden_neurons, 32, 3, 2, padding=1)
        time = int((time + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1)
        self.conv1d2 = nn.Conv1d(32, 16, 3, 2, padding=1)
        time = int((time + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1)
        self.conv1d3 = nn.Conv1d(16, 16, 3, 2, padding=1)
        time = int((time + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1)
        self.time = time
        self.fc2 = nn.Linear(16 * time, bottleneck_neurons * 2)

    def forward(self, x, session_index):
        x = torch.nn.LeakyReLU(0.2)(self.fc1[session_index](x))
        x = x.permute(1, 2, 0)
        x = torch.nn.LeakyReLU(0.2)(self.conv1d(x))
        x = torch.nn.LeakyReLU(0.2)(self.conv1d2(x))
        x = torch.nn.LeakyReLU(0.2)(self.conv1d3(x))
        x = torch.flatten(x, 1)
        x = torch.nn.LeakyReLU(0.2)(self.fc2(x))
        return x
