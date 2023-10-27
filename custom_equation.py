from numpy import *


def custom_distribution(text_equation):
    print("start")
    size_distribution = 256
    mass = zeros(size_distribution, dtype="int64")

    print(text_equation)

    if "x" not in text_equation:
        return mass

    for x in range(size_distribution):
        mass[x] = eval(text_equation)

    return mass
