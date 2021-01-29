import gym
import time
import argparse
import numpy as np
import torch
import collections

from master.Chapter06.libc import dqn_model, wrappers


def main():
    env_name = "PongNoFrameskip-v4"
    frame_per_sec = 25

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="./results/PongNoFrameskip-v4-best.dat",
        help="Model file to load",
    )
    parser.add_argument(
        "-e",
        "--env",
        default=env_name,
        help="Environment name to use, default=" + env_name,
    )
    parser.add_argument("-r", "--record", default="./video", help="Directory for video")
    parser.add_argument(
        "--no-vis", default=True, dest="vis", help="Disable visualization", action="store_false"
    )
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    print(env.unwrapped.get_action_meanings())
    if args.record:
        env = gym.wrappers.Monitor(env, args.record, force=True)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.vis:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if args.vis:
            delta = 1 / frame_per_sec - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()


if __name__ == "__main__":
    main()
