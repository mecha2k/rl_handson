from master.Chapter06.libc import dqn_model, wrappers

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

Experience = collections.namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
        )


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state = None
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu", gamma=0.99):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def main():
    DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
    MEAN_REWARD_BOUND = 19

    batch_size = 32
    sync_target_frames = 1000
    eps_decay_last_frame = 150000
    eps_start = 1.0
    eps_final = 0.01
    replay_size = 10000
    replay_start_size = 10000
    learning_rate = 1e-4

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV_NAME,
        help="Name of the environment, default=" + DEFAULT_ENV_NAME,
    )
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    print(net)

    buffer = ExperienceBuffer(replay_size)
    agent = Agent(env, buffer)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(eps_final, eps_start - frame_idx / eps_decay_last_frame)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print(
                "{:d}: done {:d} games, reward {:.3f}, "
                "eps {:.2f}, speed {:.2f} f/s".format(
                    frame_idx, len(total_rewards), m_reward, epsilon, speed
                )
            )
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env + "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < replay_start_size:
            continue

        if frame_idx % sync_target_frames == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(batch_size)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
