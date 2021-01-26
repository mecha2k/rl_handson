import random
import numpy as np
import matplotlib.pyplot as plt


def main():
    a = random.random()
    b = np.random.rand()
    print(a, b)

    nums = 100
    c = [random.random() for i in range(nums)]
    d = np.random.rand(nums)
    print(c)
    print(d)

    plt.figure(figsize=(6, 4))
    plt.hist(c, bins=10, color="r")
    plt.hist(d, bins=10, color="b")
    plt.show()


if __name__ == "__main__":
    main()
