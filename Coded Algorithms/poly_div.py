"""
Marcelo Leszynski
7/30/20

The poly_div() function is an implementation of the polynomial division algorithm 
found on page 65 of the text. The "__main__" code uses this algorithm to verify 
the answers to exercises 2.3.1a and 2.3.1b as required in exercise 2.3.3.

Note: modifications had to be made to the algorithm proposed in the text in order 
to handle cases where a given quotient was equal to zero.
"""

import sympy as sy

x,y = sy.symbols('x,y')

def poly_div(divisors, dividend, grlex = False):
    quotients = []
    remainder = 0
    p = dividend
    while p != 0:
        i = 0
        divisionoccured = False
        while i < len(divisors) and divisionoccured == False:
            if (sy.degree(p, gen=x) != 0 or sy.degree(p, gen=y) !=0) and (sy.degree(divisors[i], gen=x) != 0 or sy.degree(divisors[i], gen=y) != 0):
                if grlex:
                    tempq, tempr = sy.div(sy.LT(p, order = 'grlex'), sy.LT(divisors[i], order = 'grlex'), domain = 'ZZ') 
                else:
                    tempq, tempr = sy.div(sy.LT(p), sy.LT(divisors[i]), domain = 'ZZ') 
                if i < len(quotients):
                    quotients[i] += tempq
                else:
                    quotients.append(tempq)
                p -= tempq * divisors[i]
                p = sy.expand(p)
            if tempr == 0:
                divisionoccured = True
            else:
                i += 1
        if divisionoccured == False:
            if sy.degree(p, gen=x) != 0 or sy.degree(p, gen=y) != 0:
                if grlex:
                    remainder += sy.LT(p, order = 'grlex')
                    p -= sy.LT(p, order = 'grlex')
                    p = sy.expand(p)
                else:
                    remainder += sy.LT(p)
                    p -= sy.LT(p)
                    p = sy.expand(p)
            else:
                remainder += p
                p = 0
                break
    return quotients, remainder

if __name__ == "__main__":
    expr = x**7*y**2 + x**3*y**2 -y + 1
    fs = [x*y**2-x, x-y**3]
    fr = [x-y**3, x*y**2-x]

    temp1, temp2 = poly_div(fs, expr, True)
    print("Dividing x**7*y**2 + x**3*y**2 -y + 1 by [x*y**2-x, x-y**3] in grlex order:")
    print(temp1, temp2, "\n\n")

    temp1, temp2 = poly_div(fs, expr)
    print("Dividing x**7*y**2 + x**3*y**2 -y + 1 by [x*y**2-x, x-y**3] in lex order:")
    print(temp1, temp2, "\n\n")

    temp1, temp2 = poly_div(fr, expr, True)
    print("Dividing x**7*y**2 + x**3*y**2 -y + 1 by [x-y**3, x*y**2-x] in grlex order:")
    print(temp1, temp2, "\n\n")

    temp1, temp2 = poly_div(fr, expr)
    print("Dividing x**7*y**2 + x**3*y**2 -y + 1 by [x-y**3, x*y**2-x] in lex order:")
    print(temp1, temp2, "\n\n")