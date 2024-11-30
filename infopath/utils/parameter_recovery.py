import torch
from infopath.model_loader import load_model_and_optimizer
from infopath.config import load_training_opt, save_opt
import os
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt


def generate_and_save_data(
    save_path="datasets/ModelData",
    log_path="log_dir/a2c6ad1709e4cdb13e66119cc295059401535b3b/2024_1_16_6_56_5_chain/",
    trials=400,
    area_readout=1,
    session_name="PS000_20220404",
    save=True,
    seed=0,
    last_best="last",
):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, session_name)):
        os.mkdir(os.path.join(save_path, session_name))
    opt = load_training_opt(log_path)
    save_opt(save_path, opt)
    model = load_model_and_optimizer(opt, reload=True, last_best=last_best)[0]
    model.rsnn.temperature.data = torch.tensor(model.opt.temperature)
    model.to(model.opt.device)
    # model.rsnn._w_rec.data[model.rsnn.upsp_rec() < 1e-2] = 0

    # high_syn = model.rsnn.upsp_in() > 10
    # out_high_syn = torch.where(high_syn)[0]
    # model.rsnn._w_in.data[high_syn] = (
    #     model.rsnn.tau / 10 * (model.rsnn.thr0 - model.rsnn.v_rest)
    # )[out_high_syn]

    stims = torch.ones(trials) * 4
    # print(((model.rsnn._w_rec.data > 10e-8) * 1.0).mean())
    # model.rsnn._w_in.data[model.rsnn._w_in < 10e-4] = 0
    # model.rsnn._w_rec.data[model.rsnn._w_rec < 10e-3] = 0
    torch.manual_seed(seed)
    with torch.no_grad():
        spikes, _, _, _ = model(stims)
    if "Mechanism" in save_path or "GoNoGo" in save_path:
        a = 8 * model.timestep
        filt = model.filter_fun2(model.filter_fun1(spikes))
        tt1 = filt[:, :, model.rsnn.area_index == 0].mean(2).max((0))[0] > a
        tt2 = filt[:, :, model.rsnn.area_index == 1].mean(2).max((0))[0] > a
        tt3 = filt[:, :, model.rsnn.area_index == 2].mean(2).max((0))[0] > a
        trial_type = tt2 + tt1 * 2 + tt3 * 4
    else:
        trial_type = (
            spikes[:, :, model.rsnn.area_index == area_readout].mean((0, 2))
            / model.timestep
        ) > 13

    print("Hit rate: ", trial_type.unique(return_counts=True)[1] / trials)
    trial_df = pd.DataFrame(
        columns=pd.read_csv(
            os.path.join(opt.datapath, session_name, "trial_info")
        ).keys()[1:]
    )
    if "Mechanism" in save_path or "GoNoGo" in save_path:
        tt = {i: i for i in range(trial_type.max() + 1)}
    else:
        tt = {0: "Miss", 1: "Hit"}
    df_entry = pd.DataFrame(columns=trial_df.columns)
    for i in range(trials):
        df_entry["trial_number"] = [i]
        df_entry["trial_type"] = [tt[trial_type.cpu().numpy()[i]]]
        df_entry["trial_onset"] = [0.1 + i * model.T * model.timestep]
        df_entry["reaction_time_piezo"] = [0.1]
        df_entry["reaction_time_jaw"] = [0.1]
        df_entry["stim"] = [int(stims.numpy()[i])]
        df_entry["trial_active"] = [np.random.randint(0, 2)]
        df_entry["video_onset"] = [2]
        df_entry["video_offset"] = [0]
        trial_df = pd.concat([trial_df, df_entry], ignore_index=True)
    area_dict = {0: "area1", 1: "area2", 2: "area3"}
    if save:
        trial_df.to_csv(os.path.join(save_path, session_name, "trial_info"))
        neuron_df = pd.DataFrame(
            columns=["session", "area", "excitatory", "firing_rate", "cluster_index"]
        )
        neuron_entry = pd.DataFrame(
            columns=["session", "area", "excitatory", "firing_rate", "cluster_index"]
        )
        # dense and sparse implementation
        if True:  # spikes.unique().shape[0] == 2:
            neurons, tms = torch.where(
                spikes.cpu().permute(2, 1, 0).reshape(opt.n_units, -1)
            )
            for i in range(opt.n_units):
                n_ind = int(model.neuron_index[i].item())
                neuron_entry["session"] = [session_name]
                neuron_entry["area"] = [area_dict[model.rsnn.area_index[i].item()]]
                neuron_entry["excitatory"] = [model.rsnn.excitatory_index[i].item()]
                sptms = tms[neurons == i].numpy() * model.timestep
                # baseline firing rate
                neuron_entry["firing_rate"] = [
                    ((sptms % (model.T * model.timestep)) < 0.1).sum() / (trials * 0.1)
                ]
                np.save(
                    os.path.join(
                        save_path, session_name, "neuron_index_{}".format(n_ind)
                    ),
                    sptms,
                )
                neuron_entry["cluster_index"] = [n_ind]
                neuron_df = pd.concat([neuron_df, neuron_entry], ignore_index=True)
        else:
            for i in range(opt.n_units):
                n_ind = int(model.neuron_index[i].item())
                neuron_entry["session"] = [session_name]
                neuron_entry["area"] = [area_dict[model.rsnn.area_index[n_ind].item()]]
                neuron_entry["excitatory"] = [model.rsnn.excitatory_index[n_ind].item()]
                neuron_entry["firing_rate"] = [
                    (spikes[:, :, i].mean() / model.timestep).item()
                ]
                neuron_entry["cluster_index"] = [n_ind]
                neuron_df = pd.concat([neuron_df, neuron_entry], ignore_index=True)
                np.save(
                    os.path.join(
                        save_path, session_name, "neuron_index_{}".format(n_ind)
                    ),
                    spikes[:, :, i].T.reshape(-1).cpu().numpy(),
                )

        neuron_df = neuron_df.sort_values(by=["cluster_index"])
        neuron_df = neuron_df.reset_index(drop=True)
        neuron_df.to_csv(os.path.join(save_path, "cluster_information"))


# for seed in [0, 1, 2, 3, 4]:
#     generate_and_save_data(
#         "datasets/GoNoGo_seed{}".format(seed),
#         "log_dir/c0e11bcfc68c6ab68aa2679b53e4bef58f5b33d5/2024_3_25_10_40_37_mechanism1_2areas_cp/",
#         area_readout=1,
#         save=True,
#         seed=seed,
#         trials=800,
#     )
# generate_and_save_data(
#     "datasets/GoNoGo_nofb",
#     # "log_dir/c0e11bcfc68c6ab68aa2679b53e4bef58f5b33d5/2024_3_25_10_40_37_mechanism1_2areas_cp/",
#     "log_dir/afa49d0731e6f8c21564787e82c28a138b6fa0b6/2024_5_13_14_45_50_teacher_conf/",
#     area_readout=1,
#     save=True,
#     seed=0,
#     trials=1200,
# )
generate_and_save_data(
    "datasets/GoNoGo_nofb_seed0",
    "log_dir/1d74764c4551eef5158418ea67fbe1a5885dfdb1/2024_5_27_9_46_33_teacher_conf_block",
    # "log_dir/1bde1ac87d7cb15dc373cf094c2db105bed98319/2024_6_10_21_36_1_teacher_conf_block",
    area_readout=1,
    save=True,
    seed=0,
    trials=2000,
    last_best="best",
)
generate_and_save_data(
    "datasets/GoNoGo_withfb1_seed0",
    "log_dir/1d74764c4551eef5158418ea67fbe1a5885dfdb1/2024_5_27_9_46_33_teacher_conf",
    # "log_dir/1bde1ac87d7cb15dc373cf094c2db105bed98319/2024_6_10_21_36_0_teacher_conf",
    area_readout=1,
    save=True,
    seed=0,
    trials=2000,
    last_best="best",
)
generate_and_save_data(
    "datasets/GoNoGo_nofb_seed1",
    "log_dir/1d74764c4551eef5158418ea67fbe1a5885dfdb1/2024_5_27_9_46_33_teacher_conf_block",
    # "log_dir/1bde1ac87d7cb15dc373cf094c2db105bed98319/2024_6_10_21_36_1_teacher_conf_block",
    area_readout=1,
    save=True,
    seed=1,
    trials=2000,
    last_best="best",
)

generate_and_save_data(
    "datasets/GoNoGo_withfb1_seed1",
    "log_dir/1d74764c4551eef5158418ea67fbe1a5885dfdb1/2024_5_27_9_46_33_teacher_conf",
    # "log_dir/1bde1ac87d7cb15dc373cf094c2db105bed98319/2024_6_10_21_36_0_teacher_conf",
    area_readout=1,
    save=True,
    seed=1,
    trials=2000,
    last_best="best",
)
generate_and_save_data(
    "datasets/GoNoGo_nofb_seed2",
    "log_dir/1d74764c4551eef5158418ea67fbe1a5885dfdb1/2024_5_27_9_46_33_teacher_conf_block",
    # "log_dir/1bde1ac87d7cb15dc373cf094c2db105bed98319/2024_6_10_21_36_1_teacher_conf_block",
    area_readout=1,
    save=True,
    seed=2,
    trials=2000,
    last_best="best",
)
generate_and_save_data(
    "datasets/GoNoGo_withfb1_seed2",
    "log_dir/1d74764c4551eef5158418ea67fbe1a5885dfdb1/2024_5_27_9_46_33_teacher_conf",
    # "log_dir/1bde1ac87d7cb15dc373cf094c2db105bed98319/2024_6_10_21_36_0_teacher_conf",
    area_readout=1,
    save=True,
    seed=2,
    trials=2000,
    last_best="best",
)
# generate_and_save_data(
#     "datasets/GoNoGo_withfb2_seed1",
#     "log_dir/1d74764c4551eef5158418ea67fbe1a5885dfdb1/2024_5_27_9_46_33_teacher_conf_differential/",
#     # "log_dir/1bde1ac87d7cb15dc373cf094c2db105bed98319/2024_6_10_21_36_1_teacher_conf_differential",
#     area_readout=1,
#     save=True,
#     seed=1,
#     trials=2000,
#     last_best="best",
# )
# generate_and_save_data(
#     "datasets/Mechanism2_areas2",
#     "./log_dir/62a57e36b1fea8a2585cb2da0613280e0a5fb0d4/2024_2_8_17_9_10_mechanism2_2areas/",
#     area_readout=1,
#     save=True,
#     seed=3,
# )

# generate_and_save_data(
#     "datasets/Mechanism3",
#     "./log_dir/0909abaacfc6a68eb8e7cb3f04bc16d2ae8ef3bd/2024_2_7_17_35_8_trial/",
#     area_readout=1,
#     save=True,
# )
