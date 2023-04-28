# total timesteps = steps that env take ? each epoch?
# from torch.utils.tensorboard import SummaryWriter
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=path.basename(__file__).rstrip(".py"))
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--total-timesteps", type=int, default=25000)
    # parser.add_argument("--torch-deterministicm", type=lambda x:bool(strtobool(x)), default=True)
    parser.add_argument("--gpu", type=int, default=1, help="1: on and find suitable device automatically; 0: off")
    parser.add_argument("--video", type=int, default=0)

    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=128, help="number of steps to step for each parallel env in a rollout")
    # epoch steps = num.env * num.steps
    parser.add_argument("--anneal-lr", type=int, default=1)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--training-iter", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    parser.add_argument("--load-model", type=str, default=None)

    args = parser.parse_args()
    args.batch_size = int(args.num_steps * args.num_envs)  # i.e. batch step size (each epoch)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def make_env(env_id, run_name, idx, record_video):
    def fn():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if record_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x % 20 == 0, disable_logger=True)
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env
    return fn

# init layer params in place


def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    # Ref: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks - Saxe, A. et al. (2013).
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# def plot_rewards(episode_rewards):
#     plt.figure(1)
#     rewards = torch.tensor(episode_rewards)

#     plt.clf()
#     plt.xlabel('episode')
#     plt.ylabel('reward')
#     plt.plot(rewards.numpy())
#     if len(episode_rewards) > 50:
#         moving_average = rewards.unfold(0, 50, 1).mean(1).view(-1)
#         moving_average = torch.cat([torch.zeros(49), moving_average])
#         plt.plot(moving_average.numpy())
#     plt.pause(0.001)


# TODO: to reference v1 code
class Agent(nn.Module):
    def __init__(self, envs, load_from=None):
        super(Agent, self).__init__()
        # act = nn.Tanh()

        critic_layers = [
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), 1.0)
        ]
        self.critic = nn.Sequential(*critic_layers)
        print("Critic Network")
        print(self.critic)

        actor_layers = [
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # small gain (i.e. std) to ensure the output probability between actions are similar
            layer_init(nn.Linear(64, envs.single_action_space.n), 0.01),
        ]
        self.actor = nn.Sequential(*actor_layers)
        print("Actor Network")
        print(self.actor)

        if load_from:
            self.actor.load_state_dict(load_from["actor"])
            self.critic.load_state_dict(load_from["critic"])

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        policy_distri = Categorical(logits=logits)
        if action == None:
            action = policy_distri.sample()
        return action, policy_distri.log_prob(action), policy_distri.entropy(), self.get_value(obs)


# TODO: to reference v1 code

if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_name = f"{args.env_id}__{args.exp_name}_{args.seed}__{datetime.now().strftime('%Y-%m-%d %H:%M')}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = "cpu"
    if args.gpu == 1:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    device = torch.device(device)

    # each callback to create a sub-env; for parallel processing of env
    env_fns = [make_env(args.env_id, run_name, i, args.video)
               for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only support discrete control problem"
    obs_shape = envs.single_observation_space.shape
    num_action = envs.single_action_space.n
    print("envs.single_observation_space.shape", obs_shape)
    print("evns.single_action_space.h", num_action)

    model_saver = ModelSaver()

    load_from = None
    if (args.load_model):
        load_from = {
            "actor": model_saver.load(f"{args.load_model}-ac"),
            "critic": model_saver.load(f"{args.load_model}-cr")
        }
    agent = Agent(envs, load_from).to(device)

    # Ref: Adam: A Method for Stochastic Optimization
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    # at the end of each epoch (after update) - scheduler.step()

    # buffer init
    obs_buf = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    # not entirely sure about the dimension of this
    act_buf = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logp_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rew_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    term_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    val_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # global state init
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)  # init obs?
    next_term = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    epoch_avg_rtns = []

    # training with annealing lr
    # TODO: use scheduler to anneal lr instead
    for epoch in range(1, num_updates + 1):
        epi_rtns = []

        if args.anneal_lr:
            frac = 1 - (epoch - 1) / num_updates
            lr_ = frac * args.lr
            optimizer.param_groups[0]["lr"] = lr_

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs_buf[step] = next_obs
            term_buf[step] = next_term

            with torch.no_grad():
                act, logp, _, val = agent.get_action_and_value(next_obs)
                val_buf[step] = val.flatten()
            act_buf[step] = torch.tensor(act, dtype=torch.float32)
            logp_buf[step] = logp

            next_obs, rew, term, trun, info = envs.step(act.cpu().numpy())
            rew_buf[step] = torch.Tensor(rew).to(device).view(-1)  # to confirm
            # *** check if the env has trun (and whether to include it as episode end condition)
            next_obs, next_term = torch.tensor(next_obs).to(device), torch.tensor(np.logical_or(term, trun), dtype=torch.float32).to(device)
            # next_obs, next_term = torch.tensor(next_obs).to(device), torch.tensor(term).to(device)

            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        print(f"epoch={epoch}, global_step={global_step}, episodic_return={item['episode']['r'].item()}")
                        epi_rtns.append(item['episode']['r'].item())

        # for plotting, only record actually terminated / truncated episodes (not cut-off episodes)
        epoch_avg_rtns.append(sum(epi_rtns) / len(epi_rtns))

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            # TODO: try to substitute in OpenAI's implementation of GAE
            adv_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_term
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - term_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                # bootstrap cut off episdoes
                delta = rew_buf[t] + args.gamma * nextvalues * nextnonterminal - val_buf[t]
                adv_buf[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = adv_buf + val_buf

        b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_act = act_buf.reshape((-1,) + envs.single_action_space.shape)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv_buf.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        b_idx = np.arange(args.batch_size)
        clipfracs = []  # how often the clipping due to kl occurs; more investigations needed
        for iter in range(args.training_iter):
            np.random.shuffle(b_idx)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_idx = b_idx[start:end]

                # _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                # logratio = newlogprob - b_logprobs[mb_inds]
                # ratio = logratio.exp()
                _, logp_new, entropy, value_new = agent.get_action_and_value(b_obs[mb_idx], b_act.long()[mb_idx])
                log_ratio = logp_new - b_logp[mb_idx]
                ratio = torch.exp(log_ratio)
                # *** why is it not all ones??

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # advantage normalisation
                mb_adv = b_adv[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # compute policy loss
                # TODO: switch around the positive / negative signs
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                ones = torch.full(ratio.shape, 1.0).to("mps")

                # Value loss
                value_new = value_new.view(-1)
                v_loss_unclipped = (value_new - b_ret[mb_idx]) ** 2
                v_clipped = b_val[mb_idx] + torch.clamp(
                    value_new - b_val[mb_idx],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_ret[mb_idx]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                # normal unclipped v_loss - v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                # print("total loss", loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # TODO: can also implement at minibatch level
            # if args.target_kl is not None:
            #     if approx_kl > args.target_kl:
            #         break

        if epoch % 10 == 1:
            model_saver.save(agent.actor, agent.critic, epoch)

        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = np.var(y_true)
        # explained variables measures how good your value function is (in estimating returns)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    print("SPS:", int(global_step / (time.time() - start_time)))
    print("epoch_avg_rtns", epoch_avg_rtns)

    envs.close()

    # plotting
    plt.plot(epoch_avg_rtns)
    plt.xlabel("Epoch")
    plt.ylabel("Average Episode Return (excl. cut-off tracjectories)")
    plt.show()

    # TODO: add plotting function
    # TODO: add model saving
