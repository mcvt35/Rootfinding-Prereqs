# profiling.py
"""Python Essentials: Profiling.
Marcelo Leszynski
<Class>
04/30/20
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
from numba import jit
from matplotlib import pyplot as plt
import time

# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
        data = data[::-1]
        N = len(data)
        for i in range(1, N):
            N_i = len(data[i])
            for j in range(N_i):
                if data[i - 1][j] >= data[i - 1][j + 1]:
                    data[i][j] += data[i - 1][j]
                else:
                    data[i][j] += data[i - 1][j + 1]
    return data[N - 1][0]


# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = []
    for i, x in enumerate(prime_sieve(N)):
        primes_list.append(x)
    return primes_list
    

# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    x = x.reshape(len(x),1)
    A = A - x
    return np.argmin(np.linalg.norm(A, axis = 0))


# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    alphabet = {'\n':0, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10, 'K':11, 'L':12, 'M':13, 'N':14, 'O':15, 'P':16, 'Q':17, 'R':18, 'S':19, 'T':20, 'U':21, 'V':22, 'W':23, 'X':24, 'Y':25, 'Z':26}
    names = [sum([alphabet[letter] for letter in name]) for name in names]
    for count, name in enumerate(names, 1):
        total += count * name
    return total



# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    x = 0
    y = 0
    sum = 1
    while True:
        sum = x + y
        if sum == 0:
            sum = 1
        x = y
        y = sum
        yield sum

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    for index, number in enumerate(fibonacci(), 1):
        if len(str(number)) == N:
            print(number, len(str(number)))
            return index


# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    is_prime = [False] * 2 + [True] * (N - 1)
    for n in range(int(N**0.5 + 1.5)):
        if is_prime[n]:
            for i in range(n * n, N + 1, n):
                is_prime[i] = False
    final = [i for i, prime in enumerate(is_prime) if prime]
    for i in final:
        yield i


# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    times1 = []
    times2 = []
    times3 = []
    
    for i in range(2, 8):
        A = np.array([ [[np.random.random() for j in range(2**i)] for j in range(2**i)] ])
        time1 = time.time()
        matrix_power(A, n)
        times1.append(time.time() - time1)
        time2 = time.time()
        matrix_power_numba(A, n)
        times2.append(time.time() - time2)
        time3 = time.time()
        np.linalg.matrix_power(A, n)
        times3.append(time.time() - time3)
    plt.plot(list(range(2,8)), times1, "k")
    plt.plot(list(range(2,8)), times2, "r")
    plt.plot(list(range(2,8)), times3, "g")
    plt.show()
    raise NotImplementedError("Problem 7 Incomplete")
