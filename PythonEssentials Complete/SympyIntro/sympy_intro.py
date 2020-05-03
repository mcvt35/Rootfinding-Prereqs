# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
Marcelo Leszynski
<Class>
05/1/20
"""

import sympy as sy
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    x = sy.symbols('x')
    y = sy.symbols('y')
    return sy.Rational(2, 5) * sy.exp(x**2 - y) * sy.cosh(x + y) + sy.Rational(3, 7) * sy.log(x * y + 1)




# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    j, x, i = sy.symbols('j x i')
    prodsum = sy.product(sy.summation(j * (sy.sin(x) + sy.cos(x)), (j, i, 5)), (i, 1, 5))
    return sy.simplify(prodsum)


# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    x, n, y = sy.symbols('x n y')
    expr = sy.summation(x**n / sy.factorial(n), (n, 0, N))
    new_expr = expr.subs(x, -y**2)
    f = sy.lambdify(y, new_expr, "numpy")
    input = np.linspace(-2, 2, 100)
    plt.plot(input, f(input), "g")
    plt.plot(input, np.exp(-input**2))
    plt.show()



# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    x, y, r, theta = sy.symbols('x y r theta')
    expr = 1 - ((x**2 + y**2)**sy.Rational(7, 2) + 18 * x**5 * y - 60 * x**3 * y**3 + 18 * x * y**5) / ((x**2 + y**2)**3)
    expolar = expr.subs({x:(r * sy.cos(theta)), y:(r * sy.sin(theta))})
    expolar = sy.simplify(expolar)
    solution = sy.solve(expolar, r)[0]
    f = sy.lambdify(theta, solution, "numpy")
    domain = np.linspace(0, 2*np.pi, 200)
    plt.plot(f(domain) * np.cos(domain), f(domain) * np.sin(domain), "r")
    plt.show()


# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    x, y = sy.symbols('x y')
    A = sy.Matrix([ [x - y, x, 0], [x, x - y, x], [0, x, x - y] ])
    eigenvects = A.eigenvects()
    return {eigenvects[0][1]:eigenvects[0][2], eigenvects[1][1]:eigenvects[1][2], eigenvects[2][1]:eigenvects[2][2]}


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    x = sy.symbols('x')
    expr = 2 * x**6 - 51 * x**4 + 48 * x**3 + 312 * x**2 - 576 * x - 100
    f = sy.lambdify(x, expr, "numpy")
    diff1 = sy.diff(expr, x)
    diff2 = sy.diff(diff1, x)
    g = sy.lambdify(x, diff2, "numpy")
    critpoints = sy.solve(diff1, x)
    maxes = []
    mins = []
    for point in critpoints:
        if g(point) > 0:
            mins.append(point)
        elif g(point) < 0:
            maxes.append(point)
    domain = np.linspace(-5, 5, 100)
    plt.plot(domain, f(domain), "k")
    for num in maxes:
        plt.plot(num, f(num), 'og')
    for num in mins:
        plt.plot(num, f(num), 'or')
    plt.show()
    return set(mins), set(maxes)




# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    # Typo on pg 151
    r, p, t, phi = sy.symbols('r p theta phi')
    expr = ((p * sy.sin(phi) * sy.cos(t))**2 + (p * sy.sin(phi) * sy.sin(t))**2 + (p * sy.cos(phi))**2 )**2
    expr = sy.simplify(expr)
    Jmat = sy.Matrix([p * sy.sin(phi) * sy.cos(t), p * sy.sin(phi) * sy.sin(t), p * sy.cos(phi)])
    J = Jmat.jacobian([p, t, phi])
    det = -J.det()
    expr = expr * det
    integral = sy.integrate(sy.integrate(sy.integrate(expr, (p, 0, r)) , (t, 0, 2 * sy.pi)), (phi, 0, sy.pi))
    f = sy.lambdify(r, integral, "numpy")
    domain = np.linspace(0, 3, 100)
    plt.plot(domain, f(domain))
    plt.show()
    return f(2)
