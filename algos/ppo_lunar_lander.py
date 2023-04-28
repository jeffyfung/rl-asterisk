import numpy as np
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import gymnasium as gym
from utils.model_saver import ModelSaver
import utils.utils as utils

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# wrapper class for processed Atari env
class AtariEnv():
    def __init__(self, env_name, render_mode):
        self.env = gym.make(env_name, render_mode=render_mode)

    def process(self, env_data):
        # TODO: preprocess the observation
        # grayscale
        # resize
        # normalize
        obs = env_data[0]
        # obs = obs[25:-60, 25:-25]
        # obs = cv2.resize(
        #     obs, (self.target_height, self.target_width), interpolation=cv2.INTER_NEAREST)
        processed = (obs, *(env_data[1:]))
        return processed

    def step(self, action):
        env_data = self.env.step(action)
        return self.process(env_data)

    def reset(self):
        env_data = self.env.reset()
        return self.process(env_data)

    def get_num_action(self):
        return self.env.action_space.n

    def get_obs_dim(self):
        return self.env.observation_space.shape[0]


class Actor(nn.Module):
    # how to NOT flattten the batch size dimension when needed?
    def __init__(self, input_shape, output_shape):
        super().__init__()
        act = nn.ReLU(True)
        layers = [
            nn.Linear(8, 128), act,
            nn.Linear(128, 32), act,
            nn.Linear(32, output_shape)
        ]
        self.net = nn.Sequential(*layers)

    # a forward pass to get softmax distribution, action
    def forward(self, obs):
        obs = obs.to(device)
        logits = self.net(obs)
        policy = Categorical(logits=logits)
        action = policy.sample()
        return policy, action

    def get_log_likelihood(self, policy, action):
        return policy.log_prob(action)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        act = nn.ReLU(True)
        layers = [
            nn.Linear(8, 128), act,
            nn.Linear(128, 32), act,
            nn.Linear(32, 1)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        obs = obs.to(device)
        return self.net(obs)


# Heavily based on openai spinningup implementation
class PPOBuffer():

    def __init__(self, buffer_size, obs_dim, gamma, lam):
        self.obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(buffer_size, dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        # only for computing GAE
        self.val_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.logp_buf = np.zeros(buffer_size, dtype=np.float32)
        self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.buffer_size = buffer_size
        self.ptr, self.epi_start_index = 0, 0

    def store_step(self, obs, act, rew, val, logp):
        assert self.ptr < self.buffer_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val):
        path_slice = slice(self.epi_start_index, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        td_residual = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = utils.discount_cumsum(
            td_residual, self.gamma * self.lam)

        self.ret_buf[path_slice] = utils.discount_cumsum(rews, self.gamma)[:-1]

        self.epi_start_index = self.ptr

    def get_all(self):
        assert self.ptr == self.buffer_size
        # advantage normalisation
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return dict(
            obs=torch.as_tensor(self.obs_buf, dtype=torch.float32),
            act=torch.as_tensor(self.act_buf, dtype=torch.float32),
            ret=torch.as_tensor(self.ret_buf, dtype=torch.float32),
            adv=torch.as_tensor(self.adv_buf, dtype=torch.float32),
            logp=torch.as_tensor(self.logp_buf, dtype=torch.float32)
        )

    # called at the end of each epoch
    def reset(self):
        self.ptr = 0
        self.epi_start_index = 0


class AtariPPO():
    def __init__(self, num_steps=5000, num_epochs=100, gamma=0.99, clip_ratio=0.2,
                 actor_lr=2.5e-4, critic_lr=2.5e-4, training_iter=15,
                 lam=0.97, max_epi_len=1500, target_kl=0.01, seed=1, render_mode=None):
        self.env = AtariEnv("LunarLander-v2", render_mode)
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.clip_ratio = clip_ratio
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.training_iter = training_iter
        self.max_epi_len = max_epi_len
        self.target_kl = target_kl
        self.saver = ModelSaver()

        np.random.seed(seed)
        torch.manual_seed(seed)

        # init actor critic
        num_action = self.env.get_num_action()
        obs_shape = self.env.get_obs_dim()
        self.actor = Actor(obs_shape, num_action).to(device)
        self.critic = Critic().to(device)

        # init buffer
        self.buffer = PPOBuffer(num_steps, obs_shape, gamma, lam)

    def get_policy_loss(self, data):
        log_prob_old, obs, act, adv = data["logp"], data["obs"], data["act"], data["adv"]
        log_prob_old = log_prob_old.to(device)
        adv = adv.to(device)

        policy = self.actor(obs)[0]
        log_prob = policy.log_prob(act.to(device))
        l_ratio = torch.exp(log_prob - log_prob_old)

        clip_adv = torch.clamp(l_ratio, 1-self.clip_ratio,
                               1+self.clip_ratio) * adv
        loss_pi = -(torch.min(l_ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (log_prob_old - log_prob).mean().item()
        ent = policy.entropy().mean().item()
        clipped = l_ratio.gt(1+self.clip_ratio) | l_ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # def get_policy_loss(self, data):
    #     # use the old policy from last gradient ascent to get a more accurate loss and gradient?
    #     # but it is more computationally expensive
    #     log_prob_old, obs, act, adv = data["logp"], data["obs"], data["act"], data["adv"]
    #     adv = adv.to(device)

    #     policy = self.actor(obs)[0]
    #     log_prob = policy.log_prob(act.to(device))

    #     l_ratio = torch.exp(log_prob - log_prob_old.to(device))

    #     # double check the shapes
    #     upper_bound = torch.clamp(l_ratio, min=1 + self.clip_ratio).to(device)
    #     lower_bound = torch.clamp(l_ratio, max=1 - self.clip_ratio).to(device)

    #     loss = torch.where(adv >= 0, upper_bound, lower_bound)
    #     loss = -(loss * adv).mean()  # negative sign for gradient ascent
    #     return loss

    def get_value_loss(self, data):
        rtn, obs = data["ret"], data["obs"]
        rtn = rtn.to(device)

        loss_func = nn.HuberLoss()
        # loss_func = nn.MSELoss()
        return loss_func(self.critic(obs).squeeze(1), rtn)

    def collect_trajectories(self, epoch):
        # init state
        obs, _ = self.env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32)
        ep_len = 0
        ep = 0
        total_rew = 0
        cum_rews = []

        for step in range(self.num_steps):
            val = self.critic(obs)
            pi, act = self.actor(obs)
            logp = self.actor.get_log_likelihood(pi, act)
            next_obs, rew, term, trun, _ = self.env.step(act.item())
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
            total_rew += rew
            term = term or trun
            # print(
            #     f"Step: {step}, action: {act.item()}, reward: {rew}, cum reward: {total_rew}, term: {term}, lives: {info['lives']}")

            ep_len += 1

            self.buffer.store_step(obs, act.item(), rew,
                                   val.item(), logp.item())

            episode_max_out = ep_len == self.max_epi_len
            step_max_out = step == self.num_steps - 1

            if term or episode_max_out or step_max_out:
                final_state_v = 0
                if not term:
                    final_state_v = self.critic(next_obs).item()
                if not step_max_out:
                    obs, _ = self.env.reset()
                    obs = torch.as_tensor(obs, dtype=torch.float32)
                    ep_len = 0
                    ep += 1
                self.buffer.finish_path(final_state_v)
                print(f"reward for this episode: {total_rew}")
                cum_rews.append(total_rew)
                total_rew = 0
            else:
                obs = next_obs

        print(f"epoch {epoch} - average reward: {np.mean(cum_rews)}")

    def update(self, epoch):
        data = self.buffer.get_all()

        policy_optim = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        value_optim = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr)

        for i in range(self.training_iter):
            policy_optim.zero_grad()
            policy_loss, policy_info = self.get_policy_loss(data)
            if policy_info["kl"] > 1.5 * self.target_kl:
                print('Early stopping at iter %d due to reaching max kl.' % i)
                break

            policy_loss.backward()
            policy_optim.step()

            value_optim.zero_grad()
            value_loss = self.get_value_loss(data)
            value_loss.backward()
            value_optim.step()

        if epoch % 5 == 0:
            self.saver.save(self.actor.net, self.critic.net, epoch)

    def train(self, load_from=None):
        if load_from:
            self.actor.net.load_state_dict(self.saver.load(f"{load_from}-ac"))
            self.critic.net.load_state_dict(self.saver.load(f"{load_from}-cr"))

        for epoch in range(self.num_epochs):
            self.collect_trajectories(epoch)
            self.update(epoch)
            self.buffer.reset()

# TODO: log and plot the training process

# TODO: preprocesssing
# TODO: early stopping for policy update based on KL divergence
# TODO: experiment with different advantage function e.g. TD-residual, Q(s,a) - V(s) etc
# see the GAE paper


def main():
    algo = AtariPPO()
    algo.train()


main()
