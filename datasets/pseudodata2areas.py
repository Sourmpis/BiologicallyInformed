import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json


def make_onset_timeseries(time, length):
    x = np.zeros(length)
    x[time] = 1
    return x


def make_kernels(tau_rise, tau_fall, timestep):
    """Make the kernel to convolve the onset timeseries."""
    max_time = int(8 * tau_fall / timestep)
    time = np.arange(max_time) * timestep
    kernel = lambda x: (np.exp(-x / tau_fall) - np.exp(-x / tau_rise))
    return kernel(time)


def firing_rates(num_neurons, average, std, mode="log_normal"):
    if mode == "log_normal":
        distro = np.random.randn(num_neurons) * np.log(std) + np.log(average)
        distro = np.exp(distro)
        # we don't allow extreme firing rates
        distro[distro > 40] = 40 - np.random.rand((distro > 40).sum()) * 10
    elif mode == "normal":
        distro = np.random.randn(num_neurons) * std + average
    elif mode == "none":
        distro = np.ones(num_neurons) * average
    return distro


class PseudoData:
    def __init__(
        self,
        path,
        mod_strength=[-2, -1, 0, 1, 2],
        mod_prob=[0.05, 0.05, 0.2, 0.5, 0.2],
        num_sessions=20,
        trials_per_session=100,
        neurons_per_sess=100,
        pEI=0.8,
        length=4000,
        trial_onset=1000,
        onsets=[5, 40],
        tau_rise=5,
        tau_fall=20,
        timestep=1,
        stim_prob=False,
        firing_rates=[3, 4],
        firing_rates_std=[2, 2],
        p_trial_type=0.8,
        structure="chain",
        areas=["area1", "area2", "area3"],
    ):
        """Generate a dataset with artificial data. The main idea is that we have a dataset
        of two areas where the first is responding always to a stimulus input at onsets[0]
        and the sencond area responds with p_trial_type probability at onsets[1].

        Args:
            path (_type_): _description_
            mod_strength (list, optional): strength of modulation. Defaults to [-2, -1, 0, 1, 2].
            mod_prob (list, optional): mod_prob[i] neurons have are modulated to the stimulus with strength mod_strength[i]. Defaults to [0.05, 0.05, 0.2, 0.5, 0.2].
            variation (int, optional): variation of the onset of the second are in timesteps. Defaults to 100.
            num_sessions (int, optional): Number of sesssions to generate. Defaults to 20.
            trials_per_session (int, optional): Number of trials per session. Defaults to 100.
            neurons_per_sess (int, optional): Number of neurons per session. Defaults to 100.
            pEI (float, optional): Ratio of excitatory neuron. Defaults to 0.8.
            length (int, optional): Trial timesteps. Defaults to 4000.
            trial_onset (int, optional): Onset of the trial. Defaults to 1000.
            onsets (list, optional): onsets[i] is the onset of the stimulus in areas[i] relative to the trial_onset. Defaults to [5, 40].
            tau_rise (int, optional): rise timeconstant of the stimulus kernel. Defaults to 5.
            tau_fall (int, optional): fall timeconstant of the stimulus kernel. Defaults to 20.
            timestep (int, optional): Timestep length usually 1ms.
            firing_rates (list, optional): firing_rates[0] = mean firing rate of excitatory neurons. Defaults to [3, 4].
            firing_rates_std (list, optional): firing_rates_std[0] = mean firing rate of excitatory neurons. Defaults to [2, 2].
            p_trial_type (float, optional): probability of a hit trial(both areas active). Defaults to 0.8.
        """
        self.path = path
        self.mod_strength = mod_strength
        self.mod_prob = mod_prob
        self.num_sessions = num_sessions
        self.trials_per_session = trials_per_session
        self.neurons_per_sess = neurons_per_sess
        self.pEI = pEI
        self.length = length
        self.trial_onset = trial_onset
        self.onsets = onsets
        self.tau_rise = tau_rise
        self.tau_fall = tau_fall
        self.timestep = timestep
        self.firing_rates = firing_rates
        self.firing_rates_std = firing_rates_std
        self.p_tt = p_trial_type
        self.structure = structure
        self.areas = areas
        self.stim_prob = stim_prob
        if os.path.exists(self.path):
            print("path exists rewriting")
        else:
            os.mkdir(self.path)
        self.make_cluster_info()
        self.make_trial_info()
        self.make_spikes()

    def make_cluster_info(self):
        """Generate neuron profile"""
        all_cluster_df = pd.DataFrame(
            columns=(
                "session",
                "area",
                "excitatory",
                "firing_rate",
                "cluster_index",
            )
        )
        total_neurons = self.num_sessions * self.neurons_per_sess
        total_neurons_exc = int(total_neurons * self.pEI)
        total_neurons_inh = total_neurons - total_neurons_exc
        fr_exc = firing_rates(
            total_neurons_exc,
            self.firing_rates[0],
            self.firing_rates_std[0],
        )
        fr_inh = firing_rates(
            total_neurons_inh,
            self.firing_rates[1],
            self.firing_rates_std[0],
        )

        e_ind, i_ind = 0, 0
        # a session can have neurons from either both areas or from one of the two
        # areas_rot = [["area1", "area2"], ["area1"], ["area2"]]
        for sess in range(self.num_sessions):
            sess_cluster_df = pd.DataFrame(
                columns=(
                    "neuron_index",
                    "area",
                    "excitatory",
                    "firing_rate",
                )
            )
            # where PS stands for PseudoSession
            session_path = "PS{:03d}_20220404".format(sess)
            os.mkdir(os.path.join(self.path, session_path))
            os.mkdir(os.path.join(self.path, session_path, self.areas[0]))
            if len(self.areas) > 1:
                os.mkdir(os.path.join(self.path, session_path, self.areas[1]))
            if len(self.areas) > 2:
                os.mkdir(os.path.join(self.path, session_path, self.areas[2]))
            # generate one table per session and a global one (the second helps to construct the RSNN)
            sess_entry = pd.DataFrame(columns=sess_cluster_df.columns)
            all_entry = pd.DataFrame(columns=all_cluster_df.columns)
            # areas = areas_rot[sess % 3]
            for n in range(self.neurons_per_sess):
                exc = int(self.pEI * self.neurons_per_sess)
                inh = self.neurons_per_sess - exc
                area = (n > exc) * (n - exc) // (inh // len(self.areas)) + (
                    n < exc
                ) * n // (exc // len(self.areas))
                excitatory = n < exc
                sess_entry["neuron_index"] = [n]
                sess_entry["area"] = [self.areas[area]]
                sess_entry["excitatory"] = [excitatory]
                sess_entry["firing_rate"] = [
                    (fr_exc[e_ind] if excitatory else fr_inh[i_ind])
                ]
                all_entry["session"] = [session_path]
                all_entry["area"] = [self.areas[area]]
                all_entry["excitatory"] = [excitatory]
                all_entry["firing_rate"] = [
                    (fr_exc[e_ind] if excitatory else fr_inh[i_ind])
                ]
                all_entry["cluster_index"] = [n]
                e_ind += excitatory
                i_ind += not excitatory

                sess_cluster_df = pd.concat(
                    [sess_cluster_df, sess_entry], ignore_index=True
                )
                all_cluster_df = pd.concat(
                    [all_cluster_df, all_entry], ignore_index=True
                )
            sess_cluster_df.to_csv(
                os.path.join(self.path, all_entry["session"][0], "cluster_info")
            )
        all_cluster_df.to_csv(os.path.join(self.path, "cluster_information"))

    def make_trial_info(self):
        """Make the csv of trial info, the trial structure is the same as the DataFromVahid_expert"""

        for sess in range(self.num_sessions):
            session_path = "PS{:03d}_20220404".format(sess)
            trial_info_df = pd.DataFrame(
                columns=(
                    "trial_number",
                    "reaction_time_jaw",
                    "reaction_time_piezo",
                    "stim",
                    "trial_type",
                    "trial_active",
                    "trial_onset",
                    "jaw_trace",
                    "tongue_trace",
                    "whisker_angle",
                    "completed_trials",
                    "video_onset",
                    "video_offset",
                )
            )
            pseudo_signal_path = os.path.join(self.path, session_path, "pseudo_signal")
            os.mkdir(pseudo_signal_path)
            entry_df = pd.DataFrame(columns=trial_info_df.columns)
            for trial in range(self.trials_per_session):
                entry_df["trial_number"] = [trial]
                # the next seven keys are legacy values and not really used
                entry_df["reaction_time_jaw"] = [0.15]
                entry_df["reaction_time_piezo"] = [0.15]
                if self.stim_prob:
                    entry_df["stim"] = [4 * (np.random.rand() > 0.5)]
                else:
                    entry_df["stim"] = [4]
                entry_df["trial_active"] = [np.random.choice([0, 1], p=[0.5, 0.5])]
                entry_df["video_onset"] = [2]
                entry_df["video_offset"] = [0]
                #
                if type(self.p_tt) == list:
                    entry_df["trial_type"] = [
                        np.random.choice(
                            [i for i in range(len(self.p_tt))], p=self.p_tt
                        )
                    ]
                else:
                    entry_df["trial_type"] = [
                        np.random.choice(["Miss", "Hit"], p=[1 - self.p_tt, self.p_tt])
                    ]
                entry_df["trial_onset"] = [trial * int(self.length / 1000)]
                trial_info_df = pd.concat([trial_info_df, entry_df], ignore_index=True)
            trial_info_df.to_csv(os.path.join(self.path, session_path, "trial_info"))

    def make_spikes(self):
        """Based on the cluster_info and trial_info generate the appropriate neuronal activity for all the neurons and trials."""

        sessions = os.listdir(self.path)
        sessions = [sess for sess in sessions if sess != "cluster_information"]
        clusters = pd.read_csv(os.path.join(self.path, "cluster_information"))
        for i, sess in enumerate(sessions):
            trial_info = pd.read_csv(os.path.join(self.path, sess, "trial_info"))
            trial_types = trial_info.trial_type.values
            session_path = os.path.join(self.path, sess)
            cluster = clusters[clusters.session == sess]
            areas = cluster.area.values
            sig_areas = self.trial_prototypes()
            # area2 responds in Hit and/or Miss depending the graph structure
            # area3 responds only in Hit trials
            if self.structure == "chain":
                # in miss 50% trials the communication is broken in area2 and 50% in area3
                r = np.random.rand(trial_types.shape[0]) > 0.5
                area2_active = (trial_types == "Miss") & r
                area2_active += trial_types == "Hit"
                area3_active = trial_types == "Hit"
            if self.structure == "parallel":
                # area2 is responding independent of area3 with 50% hit rate
                area2_active = np.random.rand(trial_types.shape[0]) > 0.5
                area3_active = trial_types == "Hit"
            if self.structure == "abstract":
                area3_active = trial_types // 4 == 1
                area1_active = (trial_types // 2) % 2 == 1
                area2_active = trial_types % 2 == 1
                sig_areas[0] = (sig_areas[0].T * area1_active).T

            sig_areas[1] = (sig_areas[1].T * area2_active).T
            if len(sig_areas) > 2:
                sig_areas[2] = (sig_areas[2].T * area3_active).T

            # neurons can be positive, negative or no modulated by the stimulus
            for i in range(len(sig_areas)):
                modulation = np.random.choice(
                    self.mod_strength, cluster.shape[0], p=self.mod_prob
                )
                sig_areas[i] = sig_areas[i][..., None] @ modulation[None] + 1

            rates = np.zeros_like(sig_areas[0])
            for i in range(len(sig_areas)):
                rates[:, :, areas == f"area{i+1}"] = sig_areas[i][
                    ..., areas == f"area{i+1}"
                ]
            rates *= cluster.firing_rate.values
            threshold = np.random.rand(rates.shape[0], rates.shape[1], rates.shape[2])
            spikes = (rates * self.timestep / 1000) > threshold
            for cl in range(cluster.shape[0]):
                spike_times = np.where(spikes[:, :, cl].flatten())[0] / 1000
                np.save(os.path.join(session_path, f"neuron_index_{cl}"), spike_times)

    def trial_prototypes(self):
        """Generate the population average activity of every area for every trial."""
        # all times are in ms
        sig = make_onset_timeseries(self.onsets[0] + self.trial_onset, self.length)
        kernel1 = make_kernels(self.tau_rise, self.tau_fall, self.timestep)
        sig1 = np.convolve(sig, kernel1)[: -kernel1.shape[0] + 1]
        sig1 /= sig1.max()
        sig1 *= 3
        sigs_areas = [[] for i in range(len(self.areas))]
        for _ in range(self.trials_per_session):
            sigs_areas[0].append(sig1)
            for area in range(1, len(self.areas)):
                sig = make_onset_timeseries(
                    self.trial_onset + self.onsets[area], self.length
                )
                kernel = make_kernels(self.tau_rise, self.tau_fall, self.timestep)
                sig = np.convolve(sig, kernel)[: -kernel.shape[0] + 1]
                sigs_areas[1].append(sig / sig.max() * 3)

        sigs_areas = [np.stack(sigs_area) for sigs_area in sigs_areas]
        return sigs_areas


if __name__ == "__main__":

    conf = {
        "onsets": [4, 12],
        "tau_rise": 5,
        "tau_fall": 20,
        "firing_rates": [1.45, 2.1],
        "firing_rates_std": [3.5, 3.7],
        "p_trial_type": [0.5, 0, 0, 0.5],
        "trials_per_session": 400,
        "mod_prob": [0, 0, 0.2, 0.6, 0.2],
        "pEI": 0.8,
        "stim_prob": False,
    }

    path = f"datasets/PseudoData_2areas_abstract_gonogo_v1"
    pseudo_data = PseudoData(
        path,
        onsets=conf["onsets"],
        tau_rise=conf["tau_rise"],
        tau_fall=conf["tau_fall"],
        firing_rates=conf["firing_rates"],
        firing_rates_std=conf["firing_rates_std"],
        p_trial_type=conf["p_trial_type"],
        trials_per_session=conf["trials_per_session"],
        mod_prob=conf["mod_prob"],
        pEI=conf["pEI"],
        areas=["area1", "area2"],
        num_sessions=1,
        neurons_per_sess=500,
        structure="abstract",
        stim_prob=conf["stim_prob"],
        trial_onset=200,
    )
    json.dump(conf, open(path + "/conf.json", "w"))
