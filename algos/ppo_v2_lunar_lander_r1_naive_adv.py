import argparse
from os import path
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
import time
import matplotlib.pyplot as plt
from utils.model_saver import ModelSaver
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, LinearLR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str,
                        default=path.basename(__file__).rstrip(".py"))
    parser.add_argument("--env-id", type=str, default="LunarLander-v2")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--total-timesteps", type=int, default=50000)  # 200000

    parser.add_argument("--gpu", type=int, default=1,
                        help="1: on and find suitable device automatically; 0: off")
    parser.add_argument("--video", type=int, default=0)  # not working for now

    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1024,
                        help="number of steps to step for each parallel env in a rollout")

    parser.add_argument("--anneal-lr", type=int, default=1)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--training-iter", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.05)

    parser.add_argument("--load-model", type=str, default=None)

    args = parser.parse_args()
    # i.e. batch step size (each epoch)
    args.batch_size = int(args.num_steps * args.num_envs)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_epoch = int(args.total_timesteps // args.batch_size)
    return args


def make_env(env_id, run_name, idx, record_video):
    def fn():
        env = gym.make(env_id)  # render_mode="rgb_array"
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if record_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: x % 20 == 0, disable_logger=True)
        return env
    return fn


class Agent(nn.Module):
    def __init__(self, envs, load_from=None):
        super(Agent, self).__init__()
        act = nn.ReLU()

        # can change the layer init std to 0.01
        critic_layers = [
            Agent.initialise_network_layer(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            act,
            Agent.initialise_network_layer(nn.Linear(256, 256)),
            act,
            Agent.initialise_network_layer(nn.Linear(256, 64)),
            act,
            Agent.initialise_network_layer(nn.Linear(64, 1))
        ]
        self.critic = nn.Sequential(*critic_layers)
        print("Critic Network")
        print(self.critic)

        actor_layers = [
            Agent.initialise_network_layer(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            act,
            Agent.initialise_network_layer(nn.Linear(256, 256)),
            act,
            Agent.initialise_network_layer(nn.Linear(256, 64)),
            act,
            Agent.initialise_network_layer(
                nn.Linear(64, envs.single_action_space.n)),
        ]
        self.actor = nn.Sequential(*actor_layers)
        print("Actor Network")
        print(self.actor)

        if load_from:
            self.actor.load_state_dict(load_from["actor"])
            self.critic.load_state_dict(load_from["critic"])

    def evaluate_critic(self, obs):
        return self.critic(obs)

    # TODO: separate into 2 functions
    def evaluate_actor(self, obs, action=None):
        logits = self.actor(obs)
        policy_distri = Categorical(logits=logits)
        if action == None:
            action = policy_distri.sample()
        return action, policy_distri.log_prob(action), policy_distri.entropy()

    @staticmethod
    def initialise_network_layer(layer, std=np.sqrt(2), bias=0.0):
        # Ref: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks - Saxe, A. et al. (2013).
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias)
        return layer


class SampleBuffer():
    def __init__(self, env, num_steps, num_envs, device):
        self.buffer_width = num_steps
        self.obs_buf = torch.zeros(
            (num_steps, num_envs) + env.single_observation_space.shape).to(device)
        self.act_buf = torch.zeros(
            (num_steps, num_envs) + env.single_action_space.shape).to(device)
        self.logp_buf = torch.zeros((num_steps, num_envs)).to(device)
        self.rew_buf = torch.zeros((num_steps, num_envs)).to(device)
        self.term_buf = torch.zeros((num_steps, num_envs)).to(device)
        self.val_buf = torch.zeros((num_steps, num_envs)).to(device)
        self.adv_buf = torch.zeros((num_steps, num_envs)).to(device)
        self.rtn_buf = torch.zeros((num_steps, num_envs)).to(device)

    def put(self, step, obs, term, logp, val, act, rew):
        self.obs_buf[step] = obs
        self.act_buf[step] = act
        self.logp_buf[step] = logp
        self.rew_buf[step] = rew
        self.term_buf[step] = term
        self.val_buf[step] = val

    def compute_adv_and_rtn(self, gamma, _, final_val, final_term):
        # MC return (bootstrap when necessary) - V(s)
        # TD target - V(S) => return still uses MC return
        final_val = final_val.reshape(1, -1)
        for s in reversed(range(self.buffer_width)):
            next_step_val = final_val if s == self.buffer_width - 1 else self.val_buf[s + 1]
            next_step_term_mask = final_term if s == self.buffer_width - 1 else self.term_buf[s + 1]
            next_step_non_term_mask = 1.0 - next_step_term_mask
            td_target = self.rew_buf[s] + gamma * next_step_val * next_step_non_term_mask
            self.adv_buf[s] = td_target - self.val_buf[s]
            next_step_rtn = final_val if s == self.buffer_width - 1 else self.rtn_buf[s + 1]
            self.rtn_buf[s] = self.rew_buf[s] + gamma * next_step_rtn * next_step_non_term_mask

        # for s in reversed(range(self.buffer_width)):
        #     next_step_val = final_val if s == self.buffer_width - 1 else self.val_buf[s + 1]
        #     next_step_term_mask = final_term if s == self.buffer_width - 1 else self.term_buf[s + 1]
        #     next_step_non_term_mask = 1.0 - next_step_term_mask
        #     self.rtn_buf[s] = self.rew_buf[s] + next_step_non_term_mask * gamma * next_step_val
        # self.adv_buf = self.rtn_buf - self.val_buf

            # next_step_val = final_val if s == self.buffer_width - 1 else self.val_buf[s + 1]
            # next_step_term_mask = final_term if s == self.buffer_width - 1 else self.term_buf[s + 1]
            # next_step_non_term_mask = 1.0 - next_step_term_mask
            # delta = self.rew_buf[s] + next_step_non_term_mask * gamma * next_step_val - self.val_buf[s]
            # prev_adv = 0 if s == self.buffer_width - 1 else self.adv_buf[s + 1]
            # self.adv_buf[s] = delta + next_step_non_term_mask * gamma * lam * prev_adv
        # self.rtn_buf = self.adv_buf + self.val_buf

    def get(self, obs_shape):
        flatten_obs = self.obs_buf.reshape((-1, ) + obs_shape)
        flatten_act = self.act_buf.reshape(-1)
        flatten_logp = self.logp_buf.reshape(-1)
        flatten_adv = self.adv_buf.reshape(-1)
        flatten_rtn = self.rtn_buf.reshape(-1)
        # flatten_val = self.val_buf.reshape(-1)
        return flatten_obs, flatten_act, flatten_logp, flatten_adv, flatten_rtn


class PPO():
    def __init__(self, args):
        self.args = args
        self.model_saver = ModelSaver()
        print(self.args)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

        self.device = self.get_device()
        self.envs = self.get_env()
        self.agent = self.get_agent()
        self.buffer = SampleBuffer(
            self.envs, self.args.num_steps, self.args.num_envs, self.device)

    def get_device(self):
        device = "cpu"
        if self.args.gpu == 1:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
        return torch.device(device)

    # create multiple envs for concurrent processing

    def get_env(self):
        run_name = f"{self.args.env_id}__{self.args.exp_name}_{self.args.seed}__{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        env_fns = [make_env(self.args.env_id, run_name, i, self.args.video)
                   for i in range(self.args.num_envs)]
        envs = gym.vector.SyncVectorEnv(env_fns)
        obs_shape = envs.single_observation_space.shape
        num_action = envs.single_action_space.n
        print("obs_shape", obs_shape)
        print("num_action", num_action)
        return envs

    def get_agent(self):
        load_from = None
        if (self.args.load_model):
            load_from = {
                "actor": self.model_saver.load(f"{self.args.load_model}-ac"),
                "critic": self.model_saver.load(f"{self.args.load_model}-cr")
            }
        return Agent(self.envs, load_from).to(self.device)

    def train(self):
        def collect_trajectories(epoch, obs, term):
            epi_rtns = []

            if epoch == 1:
                obs, _ = self.envs.reset()  # seed=self.args.seed
                # torch.Tensor(obs).to(self.device)
                obs = torch.tensor(obs).to(self.device)
                term = torch.zeros(self.args.num_envs).to(self.device)

            for step in range(self.args.num_steps):
                nonlocal global_step
                global_step += self.args.num_envs

                with torch.no_grad():
                    act, logp, _ = self.agent.evaluate_actor(obs)
                    val = self.agent.evaluate_critic(obs)

                obs_prime, rew, term_prime, trun, info = self.envs.step(
                    act.cpu().numpy())

                store = {
                    "obs": obs,
                    "term": term,
                    "logp": logp,
                    "val": val.flatten(),
                    "act": torch.tensor(act, dtype=torch.float32),
                    "rew": torch.tensor(rew, dtype=torch.float32).view(-1)
                }
                self.buffer.put(step, **store)

                # rew_buf[step] = torch.Tensor(rew).to(device).view(-1)  # to confirm
                obs = torch.tensor(obs_prime).to(self.device)
                term = torch.tensor(np.logical_or(
                    term_prime, trun), dtype=torch.float32).to(self.device)

                if "final_info" in info:
                    for item in info["final_info"]:
                        if item and "episode" in item:
                            print(
                                f"epoch={epoch}, global_step={global_step}, episodic_return={item['episode']['r'].item()}")
                            epi_rtns.append(item['episode']['r'].item())

            # bootstrap cut off episdoes
            with torch.no_grad():
                bootstrapped_final_val = self.agent.evaluate_critic(obs)
            self.buffer.compute_adv_and_rtn(self.args.gamma, self.args.gae_lambda, bootstrapped_final_val, term)

            # for plotting, only record actually terminated / truncated episodes (not cut-off episodes)
            # for plotting, if a epoch contains no completed episode, copy last epoch's return
            epoch_avg_rtn = sum(
                epi_rtns) / len(epi_rtns) if len(epi_rtns) > 0 else epoch_avg_rtns[-1]
            epoch_avg_rtns.append(epoch_avg_rtn)

            return obs, term

        def update(optimiser):
            # get everything from buffer
            batch_obs, batch_act, batch_logp, batch_adv, batch_rtn = self.buffer.get(self.envs.single_observation_space.shape)

            # split into minibatches
            batch_ind = np.arange(self.args.batch_size)
            for _ in range(self.args.training_iter):
                np.random.shuffle(batch_ind)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    minibatch_ind = batch_ind[start:end]

                    _, pred_logp, entropy = self.agent.evaluate_actor(
                        batch_obs[minibatch_ind], batch_act.long()[minibatch_ind])
                    pred_val = self.agent.evaluate_critic(
                        batch_obs[minibatch_ind])
                    log_ratio = pred_logp - batch_logp[minibatch_ind]
                    ratio = torch.exp(log_ratio)

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        approx_kl = ((ratio - 1) - log_ratio).mean()

                    # advantage normalisation
                    minibatch_adv = batch_adv[minibatch_ind]
                    # adding a small constant to prevent zero division
                    minibatch_adv = (minibatch_adv - minibatch_adv.mean()) / (minibatch_adv.std() + 1e-8)

                    # compute policy loss, negated for gradient ascent
                    pg_loss1 = -minibatch_adv * ratio
                    pg_loss2 = -minibatch_adv * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    v_loss = nn.HuberLoss()(pred_val.view(-1), batch_rtn[minibatch_ind])

                    entropy_loss = entropy.mean()
                    overall_loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    optimiser.zero_grad()
                    overall_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.args.max_grad_norm)
                    optimiser.step()

                if self.args.target_kl is not None:
                    if self.args.target_kl < approx_kl:
                        print("early stopping")
                        break

        global_step = 0
        epoch_avg_rtns = []
        optimiser = optim.Adam(self.agent.parameters(),
                               lr=self.args.lr, eps=1e-5)
        scheduler = LinearLR(optimiser, start_factor=0.15)

        for epoch in range(1, self.args.num_epoch + 1):
            last_epoch_final_obs = obs if epoch > 1 else None
            last_epoch_final_term = term if epoch > 1 else None
            obs, term = collect_trajectories(
                epoch, last_epoch_final_obs, last_epoch_final_term)

            update(optimiser)

            scheduler.step()

            if epoch % 20 == 1 or epoch == self.args.num_epoch:
                self.model_saver.save(
                    self.agent.actor, self.agent.critic, epoch)

        # close the envs after training
        self.envs.close()

        return epoch_avg_rtns
        # plotting
        # plt.plot(epoch_avg_rtns)
        # plt.xlabel("Epoch")
        # plt.ylabel("Average Episode Return")
        # plt.show()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    agent_returns = []

    for _ in range(4):
        algo = PPO(args)
        agent_returns.append(algo.train())

    plt.plot(np.array(agent_returns).mean(axis=0))
    plt.xlabel("Epoch")
    plt.ylabel("Average Episode Return")
    plt.show()
