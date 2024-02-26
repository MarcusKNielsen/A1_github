import math

h = 0.001
# (4,0)
a1 = [11/(12*h**2), -14/(3*h**2), 19/(2*h**2), -26/(3*h**2), 35/(12*h**2)]
# (2,2)
a2 = [-1/(12*h**2), 4/(3*h**2), -5/(2*h**2), 4/(3*h**2), -1/(12*h**2)]


def u(x):
    return math.exp(math.cos(x))

d2u_1 = 0
d2u_2 = 0

for i in range(-4, 1):
    d2u_1 += u(i*h)*a1[i+4]

for i in range(-2, 3):
    d2u_2 += u(i*h)*a2[i+2]

print("Backward:", d2u_1)
print("Centered", d2u_2)

print("Hello, World!")
