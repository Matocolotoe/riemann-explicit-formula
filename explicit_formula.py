import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import expi
from sympy import sieve
from sympy.ntheory.generate import primepi

# Amount of points to use for plotting
linspace_points = 5000

# Number of terms at which sums should be truncated
series_terms = 25

# Maximum X coordinate for plotting
x_max = 200

# Amount of non-trivial zeros of the zeta function to use (max 100 000)
zeros_amount = 10000

def numbers_from_file(path: str) -> list[float]:
    file = open(path, "r")
    i = 0
    numbers = []
    while i < zeros_amount:
        numbers.append(float(file.readline()))
        i += 1
    file.close()
    return numbers

_mu = [0] + list(sieve.mobiusrange(1, max(series_terms, x_max + 1)))
_pi = [int(primepi(n)) for n in range(0, x_max + 1)]
_zeros = numbers_from_file("zeta_zeros.txt")

def f(t: float) -> float:
    return 1 / (t * (np.power(t, 2) - 1) * np.log(t))

def li(x: complex, power: float) -> complex:
    return expi(power * np.log(x))

def riemann_j(x: float) -> float:
    s = 0
    n = 1
    while n < series_terms:
        s += _mu[n] / n * (li(x, 1 / n) - np.log(2) + quad(f, np.power(x, 1 / n), 200)[0])
        n += 1
    return s

# Plots the prime counting function
def plot_pi() -> None:
    x_values = np.linspace(0, 200, 50000)
    plt.plot(x_values, [_pi[int(x)] for x in x_values], label = "$\\pi(x)$")
    plt.xlim(0, 200)
    plt.ylim(0, 50)
    plt.grid()
    plt.show()

# Starts the animation of Riemann's formula
def plot_animation() -> None:
    x_values = np.linspace(2, x_max, x_max * 250)
    plt.plot(x_values, [_pi[int(x)] for x in x_values], label = "$\\pi(x)$")
    x_values = np.linspace(2, x_max, linspace_points)
    _j_explicit = [riemann_j(x) for x in x_values]
    graph = plt.plot(x_values, _j_explicit, label = "$\\sum \\, \\frac {\\mu(n)} {n} J(x^{1/n})$", color = "red")[0]
    plt.legend(loc = "best")
    plt.draw()
    plt.title("Approximation de $\\pi(x)$ au premier ordre")
    plt.pause(15)
    n = 0
    while n < zeros_amount:
        for i in range(len(_j_explicit)):
            x = x_values[i]
            rho = 0.5 + _zeros[n] * 1j
            s = 0
            k = 1
            while k < series_terms:
                s += _mu[k] / k * (li(x, rho / k) + li(x, (1 - rho) / k))
                k += 1
            _j_explicit[i] -= np.real(s)
        graph.set_ydata(_j_explicit)
        plt.title("Partie imaginaire du dernier zéro : ${0:.2f}$ (Total : ".format(_zeros[n]) + str(n+1) + ")")
        plt.draw()
        if n == 0:
            plt.pause(5)
        elif n < 5:
            plt.pause(0.5)
        else:
            plt.pause(0.01)
        n += 1
    plt.pause(30)

plot_animation()
