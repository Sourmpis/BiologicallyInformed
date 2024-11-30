import json
import numpy as np
import torch
from tqdm import tqdm
from infopath.utils.functions import trial_match_template, return_trial_type
import pandas as pd


def light_generator(time, window="baseline", freq=100):
    time = torch.tensor(time)
    t = torch.arange(time.shape[0]) * 2 / 1000
    light = torch.sin(2 * t * torch.pi * freq) >= 0
    if window == "baseline":
        main = (-1 <= time) & (time <= -0.2)
        ramp = (-0.2 < time) & (time <= -0.1)
    elif window == "whisker":
        main = (-0.1 <= time) & (time <= 0.1)
        ramp = (0.1 < time) & (time <= 0.2)
    elif window == "delay":
        main = (0.2 <= time) & (time <= 0.9)
        ramp = (0.9 < time) & (time <= 1)
    elif window == "response":
        main = (1 <= time) & (time <= 2)
        ramp = (2 < time) & (time <= 2.1)
    else:
        assert False, "window name wrong"
    main = main * 1.0
    start_ramp = (ramp * 1.0).argmax()
    stop_ramp = start_ramp + ramp.sum()
    t = torch.arange(time.shape[0])
    if stop_ramp != start_ramp:
        ramp = ramp * (stop_ramp - t) / (stop_ramp - start_ramp)
        total = main + ramp
    else:
        total = main
    light = light * total
    return light, total


def light_area(light, batch_size, model, area):
    assert area < model.opt.num_areas, "wrong area"
    device = model.opt.device
    light_source = torch.zeros(
        model.opt.n_units, batch_size, light.shape[0], device=device
    )
    light = light.to(model.opt.device)
    light_source[model.rsnn.area_index == area, :, :] = light
    light_source = light_source.permute(2, 1, 0) * 20 * model.rsnn.base_thr[0]
    return light_source


def opto_effect(
    time_vector,
    filt_data,
    filt_model,
    session_info,
    model,
    stims,
    input_spikes,
    state,
    mem_noise,
    filt_data_jaw=None,
    filt_jaw=None,
    lick_detector=None,
    response_time=None,
    seed=0,
    verbose=True,
    power=0.05,
    with_dt=True,
):
    trial_types, trial_types_perc = return_trial_type(
        model,
        filt_data,
        filt_model,
        filt_data_jaw,
        filt_jaw,
        session_info,
        stims,
        lick_detector,
        response_time=response_time,
    )
    miss_p = lambda x: (x == 0).sum() / torch.isin(x, torch.tensor([0, 1])).sum()
    miss_perc = miss_p(trial_types.cpu())

    if verbose:
        print(trial_types_perc)
    fun = model.step_with_dt if with_dt else model.step

    # set the input and noise
    df = pd.DataFrame(columns=("period", "area", "miss_diff", "propabilities"))
    for opto_area in range(model.opt.num_areas):
        df_entry = {
            "period": ["off"],
            "area": [model.opt.areas[opto_area]],
            "miss_diff": [torch.tensor(0).item()],
            "propabilities": [trial_types_perc.cpu().numpy().tolist()],
        }
        df_entry = pd.DataFrame(df_entry)
        df = pd.concat([df, df_entry], ignore_index=True)
        for period in ["whisker", "delay", "response"]:
            # run network with light
            light, envelope = light_generator(time_vector, period)
            envelope = model.filter_fun1(envelope[:, None, None])[:, 0, 0]
            light = light_area(light, model.opt.batch_size, model, opto_area)
            torch.manual_seed(seed)
            if type(power) == list:
                p = power[opto_area]
            else:
                p = power
            spikes_light, _, jaw_light, _ = fun(
                input_spikes, state, mem_noise, light=light * p, dt=50
            )

            filt_model_light = model.filter_fun1(spikes_light)
            filt_jaw_light = model.filter_fun1(jaw_light)

            trial_types_light, trial_types_light_perc = return_trial_type(
                model,
                filt_data,
                filt_model_light,
                filt_data_jaw,
                filt_jaw_light,
                session_info,
                stims,
                lick_detector,
                response_time=response_time,
            )
            if verbose:
                print(opto_area, period, trial_types_light_perc)
            miss_perc_light = miss_p(trial_types_light.cpu())
            df_entry["area"] = [model.opt.areas[opto_area]]
            df_entry["period"] = [period]
            df_entry["propabilities"] = [trial_types_light_perc.tolist()]
            df_entry["miss_diff"] = [(miss_perc_light - miss_perc).cpu()]
            df_entry = pd.DataFrame(df_entry)
            df = pd.concat([df, df_entry], ignore_index=True)
    return df


def run_light_tjm1_response(
    time_vector,
    model,
    stims,
    power,
    filt_data_test,
    filt_jaw_test,
    session_info_test,
    lick_classifier,
    seed=0,
):
    light, envelope = light_generator(time_vector, "response")
    envelope = model.filter_fun1(envelope[:, None, None])[:, 0, 0]
    light = light_area(light, model.opt.batch_size, model, 5)
    mem_noise = model.rsnn.mem_noise.clone()
    trial_noise = model.rsnn.trial_noise.clone()
    torch.manual_seed(seed)
    state = model.steady_state()
    input_spikes = model.input_spikes(stims)
    model.rsnn.mem_noise = mem_noise
    model.rsnn.trial_noise = trial_noise
    with torch.no_grad():
        torch.manual_seed(seed)
        spikes_light, _, jaw_light, _ = model.step_with_dt(
            input_spikes,
            state,
            light=light * power,
            sample_mem_noise=False,
            dt=50,
        )
        filt_model_light = model.filter_fun1(spikes_light) / model.timestep
        filt_jaw_light = model.filter_fun1(jaw_light)

        trial_types_light, trial_types_light_perc = return_trial_type(
            model,
            filt_data_test,
            filt_model_light,
            filt_jaw_test,
            filt_jaw_light,
            session_info_test,
            stims,
            lick_classifier,
        )
        hr = (trial_types_light == 1).sum() / (trial_types_light < 2).sum()
        far = (trial_types_light == 3).sum() / (trial_types_light >= 2).sum()
    return hr, far


def tune_power(
    time_vector,
    model,
    stims,
    filt_data_test,
    filt_jaw_test,
    session_info_test,
    lick_classifier,
    HR_light_tjM1_response,
    seed=0,
):
    with torch.no_grad():
        hr_diff = []
        powers = [0.05, 0.07, 0.1, 0.15]
        for power in powers:
            hr, far = run_light_tjm1_response(
                time_vector,
                model,
                stims,
                power,
                filt_data_test,
                filt_jaw_test,
                session_info_test,
                lick_classifier,
                seed=seed,
            )
            hr_diff.append(hr - HR_light_tjM1_response)
        hr_diff = torch.stack(hr_diff)
        power_id = torch.argmin(hr_diff.abs())
        return powers[power_id]


def model_version(log):
    model_info = pd.read_table(
        f"log_dir/1511d6ad0aaf61b032627bf30a3a6fe85f336c24/{log}/model_info.txt",
        delimiter="      ",
    )
    model_info.columns = ["step", "name"]
    df = model_info.groupby("name").max()
    df = df.reset_index()
    df.insert(2, "step_id", (df.step.values / 396).astype("int"))
    results = json.load(
        open(
            f"log_dir/1511d6ad0aaf61b032627bf30a3a6fe85f336c24/{log}/results.json", "r"
        )
    )
    if (
        results["trial_type_accuracy"][
            df[df.name == "best_trial_type_t_trial_ratio_model.ckpt"].step_id.values[0]
        ]
    ) == 1:
        model_v = "best_trial_type_t_trial_ratio"
    else:
        model_v = "best_trial_type"
    print(max(results["trial_type_accuracy"]))
    return model_v


if __name__ == "__main__":
    from infopath.utils.functions import load_data
    from infopath.lick_classifier import prepare_classifier
    from infopath.config import load_training_opt
    from infopath.model_loader import load_model_and_optimizer

    logs_rec_groups2 = [
        "2024_4_12_16_50_39_l1across01_spiking",
        "2024_4_12_16_50_34_l1across001_spiking",
        "2024_4_12_16_50_33_l1across0001_spiking",
        "2024_4_12_16_50_33_l1across00001_spiking",
    ]
    logs_rec_groups1 = [
        "2024_4_12_16_50_41_l1across01_spiking_rec_group1",
        "2024_4_12_16_50_35_l1across001_spiking_rec_group1",
        "2024_4_12_16_50_35_l1across0001_spiking_rec_group1",
        "2024_4_12_16_50_33_l1across00001_spiking_rec_group1",
    ]
    logs_rec_groups1_dist = [
        "2024_4_14_18_40_5_l1across01_spiking_rec_group1_dist",
        "2024_4_14_18_40_5_l1across001_spiking_rec_group1_dist",
        "2024_4_14_18_40_7_l1across0001_spiking_rec_group1_dist",
        "2024_4_14_18_40_7_l1across00001_spiking_rec_group1_dist",
    ]

    for name_logs, logs in zip(
        ["groups1", "groups_dist", "groups2"],
        [logs_rec_groups1, logs_rec_groups1_dist, logs_rec_groups2],
    ):
        log_path = "log_dir/1511d6ad0aaf61b032627bf30a3a6fe85f336c24/" + logs[0]
        opt = load_training_opt(log_path)
        opt.log_path = log_path
        v = model_version(log_path.split("/")[-1])
        opt.device = "cuda"
        model = load_model_and_optimizer(opt, reload=True, last_best=v)[0]

        (
            train_spikes,
            train_jaw,
            session_info_train,
            test_spikes,
            test_jaw,
            session_info_test,
        ) = load_data(model)
        filt_jaw_train = model.filter_fun1(train_jaw)
        filt_jaw_test = model.filter_fun1(test_jaw)
        data_jaw = train_jaw.clone()
        lick_classifier = prepare_classifier(
            model.filter_fun2(filt_jaw_train),
            model.filter_fun2(filt_jaw_test),
            session_info_train,
            session_info_test,
            opt.device,
        )
        sparsity_across_groups = []
        for log in logs:
            log_path = "log_dir/1511d6ad0aaf61b032627bf30a3a6fe85f336c24/" + log
            opt = load_training_opt(log_path)
            version = model_version(log)
            model = load_model_and_optimizer(opt, reload=True, last_best=version)[0]
            off_diag = model.rsnn._w_rec[:, model.rsnn.off_diag]
            sp_across = ((off_diag > 1e-5).sum() / off_diag.numel()).item()
            sparsity_across_groups.append(sp_across)
            time_vector = np.arange(model.T) * 0.002 - 0.15
            model.rsnn.light_neuron = (1 - model.rsnn.excitatory_index.long()).bool()
            with torch.no_grad():
                dfs = []
                seeds = np.arange(0, 9)
                batch_size, trials = 200, 200
                model.opt.batch_size = batch_size
                for seed in tqdm(seeds):
                    torch.manual_seed(seed)
                    stims = torch.randint(2, size=(trials,)).to(model.opt.device)
                    state = model.steady_state()
                    input_spikes = model.input_spikes(stims)
                    model.rsnn.sample_mem_noise(model.T, trials)
                    mem_noise = model.rsnn.mem_noise.clone()
                    model.rsnn.sample_trial_noise(trials)
                    torch.manual_seed(seed)
                    spikes, voltages, jaw, _ = model.step_with_dt(
                        input_spikes, state, mem_noise, dt=50
                    )
                    filt_model = model.filter_fun1(spikes)
                    del spikes
                    torch.cuda.empty_cache()
                    df = opto_effect(
                        time_vector,
                        model.filter_fun1(train_spikes),
                        filt_model,
                        session_info_train,
                        model,
                        stims,
                        input_spikes,
                        state,
                        mem_noise,
                        filt_data_jaw=model.filter_fun1(train_jaw),
                        filt_jaw=model.filter_fun1(jaw),
                        lick_detector=lick_classifier,
                        seed=seed,
                        verbose=True,
                        power=0.2,
                    )
                    dfs.append(df)

            dfs = pd.concat(dfs, ignore_index=True)
            dfs["session"] = dfs.index.values // 24
            dfs.to_csv(f"opto_effect_{log}.csv")
        json.dump(
            sparsity_across_groups,
            open(f"sparsity_{name_logs}.json", "w"),
        )
