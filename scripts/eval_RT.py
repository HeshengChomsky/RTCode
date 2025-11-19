import os
import random
import time

import d4rl
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from decision_transformer.model import RTDecisionTransformer
from decision_transformer.utils import (
    rt_evaluate,
    RTTrajectoryDataset,
    get_d4rl_dataset_stats,
    get_d4rl_normalized_score,
    parse,
    base_parse,
    random_trajectory_from_dataset,
    edt_augment_dataset,
    VAE
)
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from scripts.train_vae import TripletDataset



 
def model_test(args):

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.dataset  # medium / medium-replay / medium-expert
    rtg_scale = args.rtg_scale  # normalize returns to go
    num_bin = args.num_bin
    top_percentile = args.top_percentile
    dt_mask = args.dt_mask
    expert_weight = args.expert_weight
    exp_loss_weight = args.exp_loss_weight

    eval_dataset = args.dataset         # medium / medium-replay / medium-expert
    eval_rtg_scale = args.rtg_scale     # normalize returns to go

    if args.env == "walker2d":
        # env_name = "Walker2d-v2"
        rtg_target = 5000
        env_d4rl_name = f"walker2d-{dataset}-v2"
        env_name = env_d4rl_name

    elif args.env == "halfcheetah":
        # env_name = "HalfCheetah-v2"
        rtg_target = 6000
        env_d4rl_name = f"halfcheetah-{dataset}-v2"
        env_name = env_d4rl_name

    elif args.env == "hopper":
        # env_name = "Hopper-v2"
        rtg_target = 3600
        env_d4rl_name = f"hopper-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "ant":
        # env_name = "Hopper-v2"
        rtg_target = 3600 # the value does not really matter in our method
        env_d4rl_name = f"ant-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "antmaze-umaze":
        rtg_target = 3600 # the value does not really matter in our method
        env_d4rl_name = f"antmaze-umaze-{dataset}-v2" if dataset == "diverse" else "antmaze-umaze-v2"
        env_name = env_d4rl_name
    else:
        raise NotImplementedError

    render = args.render                # render the env frames

    num_eval_ep = args.num_eval_ep         # num of evaluation episodes
    max_eval_ep_len = args.max_eval_ep_len # max len of one episode

    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter
    mgdt_sampling = args.mgdt_sampling


    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability
    expectile = args.expectile
    rs_steps = args.rs_steps
    state_loss_weight = args.state_loss_weight
    rs_ratio = args.rs_ratio
    real_rtg = args.real_rtg
    data_ratio = args.data_ratio

    eval_chk_pt_dir = args.chk_pt_dir

    eval_chk_pt_name = args.chk_pt_name
    eval_chk_pt_list = [eval_chk_pt_name]

    device = torch.device(args.device)
    print("device set to: ", device)

    env_data_stats = get_d4rl_dataset_stats(env_d4rl_name)
    state_mean = np.array(env_data_stats['state_mean'])
    state_std = np.array(env_data_stats['state_std'])

    env = gym.make(env_d4rl_name)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load data from this file
    dataset_path_u = os.path.join(args.dataset_dir, f"{env_d4rl_name}.pkl")
    traj_dataset_u = RTTrajectoryDataset(
        dataset_path_u, context_len, rtg_scale, data_ratio=data_ratio
    )
    agument_numbers=int(len(traj_dataset_u.trajectories)*0.3)
    td = TripletDataset(traj_dataset_u)
    dl = DataLoader(td, batch_size=1, shuffle=True, drop_last=True)
    input_dim = td[0].shape[0]
    vae = VAE(input_dim, args.latent_dim, args.hidden_dim).to(device)
    vae_path=os.path.join(args.dataset_dir, f"{env_d4rl_name}.pt")
    vae = torch.load(vae_path, map_location=device)
    err_path=os.path.join(args.dataset_dir, f"{env_d4rl_name}.npz")
    err=np.load(err_path)
    max_err=err
    print(max_err)

    for eval_chk_pt_name in eval_chk_pt_list:
        model = RTDecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=embed_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=dropout_p,
            env_name=env_name,
            num_bin=num_bin,
            dt_mask=dt_mask,
            rtg_scale=rtg_scale,
            real_rtg=real_rtg,
        ).to(device)

        eval_chk_pt_path = os.path.join(eval_chk_pt_dir, eval_chk_pt_name)

        # load checkpoint
        model.load_state_dict(torch.load(eval_chk_pt_path, map_location=device))

        print("model loaded from: " + eval_chk_pt_path)

        augment_dataset = RT_augment_dataset(
            model,
            device,
            context_len,
            traj_dataset_u,
            agument_numbers,
            rtg_target,
            rtg_scale,
            top_percentile=top_percentile,
            expert_weight=expert_weight,
            num_bin=num_bin,
            env_name=env_name,
            mgdt_sampling=True,
            rs_steps=rs_steps,
            rs_ratio=rs_ratio,
            real_rtg=real_rtg,
            heuristic=args.heuristic,
            heuristic_delta=args.heuristic_delta,
            vae_model=vae,
            max_recon_error=max_err
        )

        print("augment_dataset size: ", len(augment_dataset))

if __name__ == "__main__":

    args = parse()
    model_test(args)
