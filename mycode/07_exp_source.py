import random
import collections
import numpy as np

from collections import namedtuple, deque


# one single experience step
Experience = namedtuple("Experience", ["state", "action", "reward", "done"])


class ExperienceSource:
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def __iter__(self):
        n = 0
        sequence = [1, 3, 4, 6, 87]
        for elem in sequence:
            yield n, elem
            n += 1


def my_enum(sequence, start=0):
    n = start
    for elem in sequence:
        yield n, elem
        n += 1


def main():
    seasons = ["spring", "summer", "autumn", "winter"]
    print(my_enum(seasons))
    print(list(my_enum(seasons, start=1)))

    for idx, val in my_enum(seasons):
        print(idx, val)

    env = None
    agent = None
    exp_source = ExperienceSource(env=env, agent=agent, steps_count=2)
    for idx, exp in enumerate(exp_source):
        if idx > 15:
            break
        print(idx, exp)

    print("-----")
    for exp in exp_source:
        print(exp)


if __name__ == "__main__":
    main()
