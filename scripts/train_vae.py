from ast import arg
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from decision_transformer.utils import EDTTrajectoryDataset
from decision_transformer.utils import VAE
import os
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, dt_dataset):
        self.dt = dt_dataset
        self.indices = []
        for i, traj in enumerate(self.dt.trajectories):
            T = traj["observations"].shape[0]
            for t in range(T):
                self.indices.append((i, t))
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        i, t = self.indices[idx]
        traj = self.dt.trajectories[i]
        s = torch.from_numpy(traj["observations"][t]).float()
        a = torch.from_numpy(traj["actions"][t]).float()
        sp = torch.from_numpy(traj["next_observations"][t]).float()
        x = torch.cat([s, a, sp], dim=-1)
        return x

def train_vae(dataset_path, context_len, rtg_scale, latent_dim=32, hidden_dim=256, batch_size=512, lr=1e-3, epochs=50, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt = EDTTrajectoryDataset(dataset_path, context_len, rtg_scale)
    td = TripletDataset(dt)
    dl = DataLoader(td, batch_size=batch_size, shuffle=True, drop_last=True)
    input_dim = td[0].shape[0]
    model = VAE(input_dim, latent_dim, hidden_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x in dl:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, recon_loss, kl = model.loss(recon, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    dl_eval = DataLoader(td, batch_size=batch_size, shuffle=False, drop_last=False)
    max_recon_error = 0.0
    with torch.no_grad():
        for x in dl_eval:
            x = x.to(device)
            recon, _, _ = model(x)
            err = torch.mean((recon - x) ** 2, dim=1)
            batch_max = torch.max(err).item()
            if batch_max > max_recon_error:
                max_recon_error = batch_max
    print(f"max_recon_error: {max_recon_error}")
    model_save_path = os.path.join(args.log_dir, f"{args.env}_{args.dataset}_vae.pt")
    torch.save(model, model_save_path)
    error_path = os.path.join(args.log_dir, f"{args.env}_{args.dataset}_error.npz")
    np.save(error_path, max_recon_error)
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/")
    parser.add_argument("--context_len", type=int, default=20)
    parser.add_argument("--rtg_scale", type=float, default=1000.0)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--env", type=str, default='hopper')
    parser.add_argument("--dataset", type=str, default="medium-replay")
    parser.add_argument("--max_eval_ep_len", type=int, default=1000)
    parser.add_argument("--num_eval_ep", type=int, default=20)
    parser.add_argument("--dataset_dir", type=str, default="data/")
    parser.add_argument("--log_dir", type=str, default="dt_runs/")
    args = parser.parse_args()

    dataset = args.dataset

    if args.env == "walker2d":
        rtg_target = 5000
        env_d4rl_name = f"walker2d-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "halfcheetah":
        rtg_target = 6000
        env_d4rl_name = f"halfcheetah-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "hopper":
        rtg_target = 3600
        env_d4rl_name = f"hopper-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "ant":
        rtg_target = 3600
        env_d4rl_name = f"ant-{dataset}-v2"
        env_name = env_d4rl_name
    else:
        raise NotImplementedError

    dataset_path_u = os.path.join(args.dataset_dir, f"{env_d4rl_name}.pkl")
    train_vae(
        dataset_path=dataset_path_u,
        context_len=args.context_len,
        rtg_scale=args.rtg_scale,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
    )