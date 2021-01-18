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
        history = model.fit(obs_v, acts_v, epochs=1)
        loss = history.history["loss"]
        print(f"{iter_no}: loss={loss:.2f}, rw_mean={reward_m:.1f}, rw_bound={reward_b:.1f}")
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break

    writer.close()


# def main():
#     env = gym.make("CartPole-v0")
#     env = gym.wrappers.Monitor(env, directory="mon", force=True)
#     obs_size = env.observation_space.shape[0]
#     n_actions = env.action_space.n
#
#     net = Network(obs_size, HIDDEN_SIZE, n_actions)
#     objective = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(params=net.parameters(), lr=0.01)
#     writer = SummaryWriter(comment="-cartpole")
#
#     for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
#         obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
#         optimizer.zero_grad()
#         action_scores_v = net(obs_v)
#         loss_v = objective(action_scores_v, acts_v)
#         loss_v.backward()
#         optimizer.step()
#         print(
#             "%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f"
#             % (iter_no, loss_v.item(), reward_m, reward_b)
#         )
#         writer.add_scalar("loss", loss_v.item(), iter_no)
#         writer.add_scalar("reward_bound", reward_b, iter_no)
#         writer.add_scalar("reward_mean", reward_m, iter_no)
#         if reward_m > 199:
#             print("Solved!")
#             break
#     writer.close()
#
#

# import numpy as np
# import os
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Activation, Dense, Dropout
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.datasets import mnist
# import matplotlib.pyplot as plt
# import time
#
# # os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# unique, counts = np.unique(y_train, return_counts=True)
# print("Train labels: ", dict(zip(unique, counts)))
#
# num_labels = len(np.unique(y_train))
# start_time = time.time()
#
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
#
# image_size = x_train.shape[1]
# x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
# x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255
#
# model = Sequential()
# model.add(
#     Conv2D(filters=64, kernel_size=3, activation="relu", input_shape=(image_size, image_size, 1))
# )
# model.add(MaxPooling2D(2))
# model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
# model.add(MaxPooling2D(2))
# model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
# model.add(Flatten())
# model.add(Dropout(0.2))
# model.add(Dense(num_labels))
# model.add(Activation("softmax"))
# model.summary()
#
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=1, batch_size=128)
#
# loss, acc = model.evaluate(x_test, y_test, batch_size=128)
# print("\nTest accuracy: %.1f%%" % (100.0 * acc))
# print(f"Elapsed time: {time.time()-start_time} sec")
#
# predictions = model.predict(x_test)
#
# # sample 25 mnist digits from train dataset
# indexes = np.random.randint(0, x_test.shape[0], size=25)
# images = x_test[indexes]
# labels = y_test[indexes]
#
# # plot the 25 mnist digits
# plt.figure(figsize=(10, 10))
# plt.rc("font", size=10)
# for i in range(len(indexes)):
#     plt.subplot(5, 5, i + 1)
#     image = images[i]
#     plt.imshow(image, cmap="gray")
#     plt.title(f"Label:{np.argmax(labels[i])}, Predict:{np.argmax(predictions[indexes[i]])}")
#     plt.axis("off")
#
# plt.savefig("cnn-mnist.png")
# plt.show()
# plt.close("all")
#


if __name__ == "__main__":
    main()
