from numpy import *


def custom_distribution(text_equation):
    print("start")
    size_distribution = 256
    mass = zeros(size_distribution, dtype="int64")

    print(text_equation)

    if "x" not in text_equation:
        return mass

    for x in range(size_distribution):
        val = eval(text_equation)
        if val < 0:
            val = 0
        if val >= 10000:
            val = 10000

        mass[x] = val

    return mass
