import numpy as np
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import gymnasium as gym
import utility.utils as utils
import cv2

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# wrapper class for processed Atari env
class AtariEnv():
    def __init__(self, env_name):
        self.env = gym.make(env_name)  # render_mode="human"
        self.target_height = 84
        self.target_width = 84

    def process(self, env_data):
        # TODO: preprocess the observation
        # grayscale
        # resize
        # stack 4 frames
        # normalize
        # state = obs[0]
        # info = obs[1]
        obs = env_data[0]
        # obs = np.average(obs, axis=2)
        obs = obs[25:-60, 25:-25]
        obs = cv2.resize(
            obs, (self.target_height, self.target_width), interpolation=cv2.INTER_NEAREST)
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

    def get_obs_shape(self):
        return (self.target_height, self.target_width, 3)


class Actor(nn.Module):
    # how to NOT flattten the batch size dimension when needed?
    def __init__(self, input_shape, output_shape):
        super().__init__()
        # to confirm the parameter shapes and construction of Conv2D
        # TODO: implement CNN
        act = nn.ReLU(True)
        layers = [
            nn.Conv2d(3, 32, kernel_size=8, stride=4), act,
            nn.Conv2d(32, 64, kernel_size=4, stride=2), act,
            nn.Conv2d(64, 32, kernel_size=4, stride=2), act,
            nn.Flatten(),
            nn.Linear(288, 32), act,
            nn.Linear(32, output_shape)
        ]
        # layers = [
        #     nn.Flatten(),
        #     nn.Linear(np.prod(input_shape), 32), act,
        #     nn.Linear(32, 32), act,
        #     nn.Linear(32, output_shape)
        # ]
        self.net = nn.Sequential(*layers)

    # a forward pass to get softmax distribution, action
    def forward(self, obs):
        obs = obs.to(device)
        obs = torch.movedim(obs, 3, 1)
        logits = self.net(obs)
        policy = Categorical(logits=logits)
        action = policy.sample()
        return policy, action

    def get_log_likelihood(self, policy, action):
        return policy.log_prob(action)


class Critic(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        act = nn.ReLU(True)
        layers = [
            nn.Conv2d(3, 32, kernel_size=8, stride=4), act,
            nn.Conv2d(32, 64, kernel_size=4, stride=2), act,
            nn.Conv2d(64, 32, kernel_size=4, stride=2), act,
            nn.Flatten(),
            nn.Linear(288, 32), act,
            nn.Linear(32, 1)
        ]
        # layers = [
        #     nn.Flatten(),
        #     nn.Linear(np.prod(input_shape), 32), act,
        #     nn.Linear(32, 32), act,
        #     nn.Linear(32, 1)
        # ]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        obs = obs.to(device)
        obs = torch.movedim(obs, 3, 1)
        # obs = obs.to(device)
        values = self.net(obs)
        return values


# Heavily based on openai spinningup implementation
class PPOBuffer():
    # use a deque to manage the buffer?
    # set a max size of butter
    # what to store in the buffer?
    # obvs (stacked?), action, reward,

    # obs, act, ret, adv, logp
    def __init__(self, buffer_size, obs_shape, num_action, gamma, lam):
        self.obs_buf = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        # self.act_buf = np.zeros((buffer_size, num_action), dtype=np.float32)
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

    # keep this or not?
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

    # reset the buffer
    # called after each epoch
    # move the ptr to the beginning of the buffer
    def reset(self):
        self.ptr = 0
        self.epi_start_index = 0


class AtariPPO():
    # change nuback to 4000
    def __init__(self, num_steps=4000, num_epochs=30, gamma=0.99, clip_ratio=0.2,
                 actor_lr=3e-4, critic_lr=1e-3, training_iter=100,
                 lam=0.97, max_epi_len=1000, seed=1):
        self.env = AtariEnv('ALE/Asterix-v5')
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.clip_ratio = clip_ratio
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.training_iter = training_iter
        self.max_epi_len = max_epi_len

        np.random.seed(seed)
        torch.manual_seed(seed)

        # init actor critic
        num_action = self.env.get_num_action()
        obs_shape = self.env.get_obs_shape()
        self.actor = Actor(obs_shape, num_action).to(device)
        self.critic = Critic(obs_shape).to(device)

        # init buffer
        self.buffer = PPOBuffer(num_steps, obs_shape, num_action, gamma, lam)

    def get_policy_loss(self, data):
        # use the old policy from last gradient ascent to get a more accurate loss and gradient?
        # but it is more computationally expensive
        log_prob_old, obs, act, adv = data["logp"], data["obs"], data["act"], data["adv"]
        adv = adv.to(device)

        policy = self.actor(obs)[0]
        log_prob = policy.log_prob(act.to(device))

        l_ratio = torch.exp(log_prob - log_prob_old.to(device))

        # double check the shapes
        upper_bound = torch.clamp(l_ratio, min=1 + self.clip_ratio).to(device)
        lower_bound = torch.clamp(l_ratio, max=1 - self.clip_ratio).to(device)

        loss = torch.where(adv >= 0, upper_bound, lower_bound)
        loss = -(loss * adv).mean()  # negative sign for gradient ascent
        return loss

    def get_value_loss(self, data):
        rtn, obs = data["ret"], data["obs"]
        rtn = rtn.to(device)

        # alternative: huber loss
        # loss_func = nn.HuberLoss()
        loss_func = nn.MSELoss()
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
            # may need to get logp to calculate loss
            val = self.critic(obs.unsqueeze(0))
            pi, act = self.actor(obs.unsqueeze(0))
            logp = self.actor.get_log_likelihood(pi, act)
            next_obs, rew, term, _, info = self.env.step(act.item())
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
            total_rew += rew
            assert term == False
            # print(
            #     f"Step: {step}, action: {act.item()}, reward: {rew}, cum reward: {total_rew}, term: {term}, lives: {info['lives']}")

            ep_len += 1

            # double check the arguments
            self.buffer.store_step(obs, act.item(), rew,
                                   val.item(), logp.item())

            term = info["lives"] < 3
            episode_max_out = ep_len == self.max_epi_len
            step_max_out = step == self.num_steps - 1

            if term or episode_max_out or step_max_out:
                final_state_v = 0
                if not term:
                    final_state_v = self.critic(next_obs.unsqueeze(0)).item()
                if not step_max_out:
                    obs, _ = self.env.reset()
                    obs = torch.as_tensor(obs, dtype=torch.float32)
                    ep_len = 0
                    ep += 1
                self.buffer.finish_path(final_state_v)
                # print(
                #     f"epoch {epoch} | episode {ep} - total_reward: {total_rew}")
                cum_rews.append(total_rew)
                total_rew = 0
            else:
                obs = next_obs

        print(f"epoch {epoch} - average reward: {np.mean(cum_rews)}")

    def update(self):
        # TODO: early stopping for policy update based on KL divergence
        data = self.buffer.get_all()

        policy_optim = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        value_optim = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr)

        for _ in range(self.training_iter):
            policy_optim.zero_grad()
            policy_loss = self.get_policy_loss(data)
            policy_loss.backward()
            policy_optim.step()

            value_optim.zero_grad()
            value_loss = self.get_value_loss(data)
            value_loss.backward()
            value_optim.step()

    def train(self):
        for epoch in range(self.num_epochs):
            self.collect_trajectories(epoch)
            self.update()
            self.buffer.reset()


# TODO: implement stack frames
# TODO: log and plot the training process

# TODO: preprocesssing
# TODO: use CNN for actor critic networks
# TODO: early stopping for policy update based on KL divergence
# TODO: experiment with different advantage function e.g. TD-residual, Q(s,a) - V(s) etc
# see the GAE paper

# TODO: use GPU
# TODO: save model


def main():
    # seed = np.random.randint(0, 1000)
    algo = AtariPPO()
    algo.train()


main()
