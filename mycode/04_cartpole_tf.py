import gym
import numpy as np

from collections import namedtuple
from tensorboardX import SummaryWriter
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

Episode = namedtuple("Episode", field_names=["reward", "steps"])
EpisodeStep = namedtuple("EpisodeStep", field_names=["observation", "action"])


def iterate_batches(env, model, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    while True:
        obs_v = np.array(obs).reshape(-1, len(obs))
        act_probs = model.predict(obs_v)[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    return train_obs, train_act, reward_bound, reward_mean


def main():
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = models.Sequential()
    model.add(layers.Dense(HIDDEN_SIZE, activation="relu", input_shape=(obs_size,)))
    model.add(layers.Dense(n_actions, activation="softmax"))
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, model, BATCH_SIZE)):
        obs, acts, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        obs_v = np.array(obs).reshape(-1, len(obs[0]))
        acts_v = np.array(list(map(lambda x: [0, 1] if x == 1 else [1, 0], acts)))
        history = model.fit(obs_v, acts_v, epochs=10, verbose=0)
        loss = history.history["loss"]
        print(f"{iter_no}: loss={loss[0]:.2f}, rw_mean={reward_m:.1f}, rw_bound={reward_b:.1f}")
        writer.add_scalar("loss", loss[0], iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break

    writer.close()


if __name__ == "__main__":
    main()
