from sympy import symbols, Eq, solve

# zero derivative stencil

# Define the symbols
am2, am1, a0 = symbols('am2 am1 a0')

# Define the equations
eq0 = Eq( am2+am1+a0 , 1)
eq1 = Eq( -am2*(3/2)-am1*(1/2)+a0*(1/2) , 0)
eq2 = Eq(  am2*(3/2)**2/2+am1*(1/2)**2/2+a0*(1/2)**2/2 , 0)

# Solve the system of equations
solution = solve((eq0, eq1, eq2), (am2, am1, a0))
solution




