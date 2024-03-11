from sympy import symbols, Eq, solve

# Define the symbols
a0, a1, a2, a3, a4, h = symbols('a0 a1 a2 a3 a4 h')

# Define the equations
eq0 = Eq( a0+a1+a2+a3+a4 , 0) 
eq1 = Eq( a4*4+a3*3+a2*2+a1 , 0) 
eq2 = Eq( (a4*(-4)**2/2 + a3*(-3)**2/2 + a2*(-2)**2/2 + a1/2) * h**2 - 1, 0) 
eq3 = Eq( a4*(-4)**3/3 + a3*(-3)**3/3 + a2*(-2)**3/3 - a1/3 , 0) 
eq4 = Eq( a4*(-4)**4/4 + a3*(-3)**4/4 + a2*(-2)**4/4 + a1/4, 0) 

# Solve the system of equations
solution = solve((eq0, eq1, eq2, eq3, eq4), (a0, a1, a2, a3, a4))
solution

#%%

from sympy import symbols, Eq, solve

# second derivative centered stencil

# Define the symbols
am2, am1, a0, ap1, ap2, h = symbols('am2 am1 a0 ap1 ap2 h')

# Define the equations
eq0 = Eq( am2+am1+a0+ap1+ap2 , 0)
eq1 = Eq( -2*am2-am1+ap1+2*ap2 , 0)
eq2 = Eq( (am2*(-2)**2/2 + am1*(-1)**2/2 + ap1*1**2/2 + ap2*2**2/2) * h**2 - 1, 0)
eq3 = Eq( am2*(-2)**3 + am1*(-1)**3 + ap1*1**3 + ap2*2**3 , 0)
eq4 = Eq( am2*(-2)**4 + am1*(-1)**4 + ap1*1**4 + ap2*2**4 , 0)

# Solve the system of equations
solution = solve((eq0, eq1, eq2, eq3, eq4), (am2, am1, a0, ap1, ap2))
solution