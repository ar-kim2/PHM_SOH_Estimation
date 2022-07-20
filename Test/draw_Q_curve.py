import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

x=[(490-354), (510-285), (550-399), (500-289), (550-404), (520-278), (550-389)]
y=[15,48,26,35,19,58,14]

x2=[(575-200),(575-300),(575-400),(575-500)]
y2=[54,48,24,2]


xx = [1, 2, 4, 4.01]
yy = [1, 1.019, 1.136, 1.137]

# x3 = [0.11, 0.22, 0.55, 1.1, 1.65, 2.2]
# y3 = [(4.2-4.173)/0.11, (4.2-4.159)/0.22, (4.2-4.117)/0.55, (4.2-4.047)/1.1, (4.2-4.002)/1.65, (4.2-3.942)/2.2]

x3 = [0.11, 0.22, 0.55, 1.1, 1.65, 2.2]
#y3 = [1.276, 1.251,1.16, 1, 0.781, 0.534]  # Filter 30

#y3 = [1.3, 1.275,1.173, 1, 0.773, 0.521]  # Filter 10

#y3 = [1.318, 1.29,1.181, 1, 0.767, 0.5135]  # Filter x


x33 = [0.11, 0.22, 0.55, 1.1, 1.65, 2.2]
y33 = [0, 82.84,  379.38, 1047.74, 1911.11, 2733.38]  # Filter 30# Filter 10



# x44 = [0.994225393, 0.969747282, 0.935678128, 0.920712026, 0.892634256, 0.883412974, 0.876999179, 0.864482171, \
# 0.855468239, 0.848750863, 0.844036974, 0.866082068, 0.858177932, 0.852697897, 0.851520453, 0.832010137, \
# 0.823571726, 0.821441427, 0.813640444, 0.809064205, 0.797459316, 0.791174419, 0.778312726]


x44 = [0.994225393, 0.969747282, 0.935678128, 0.920712026, 0.892634256, 0.883412974, 0.864482171, 0.855468239,\
0.848750863, 0.823571726, 0.821441427, 0.815702017, 0.809064205, 0.791174419, 0.778312726]


#
# y44 = [0.020012778, 0.022153201, 0.028775677, 0.030320961, 0.030761191, 0.032266202, 0.030183996, 0.033869295, \
# 0.034638267, 0.034960943, 0.038416369, 0.04406623, 0.044272459, 0.044899764, 0.041412744, 0.061558432, \
# 0.069681334, 0.076942515, 0.097101031, 0.094907156, 0.110599576, 0.107109935, 0.107058422]

y44 = [0.174518931, 0.177533223, 0.185371967, 0.187451542, 0.188894148, 0.190728359, 0.193007282, 0.194098051,
0.194660537, 0.230279823, 0.237617056, 0.256404971, 0.256023564, 0.268865008, 0.269272658]




# y3 = [1.311, 1.282,1.177, 1, 0.769, 0.5148]  # Filter 5

x4 = [0, 3]
y4 = [5, 30]

x_idx = np.linspace(0, 500, 500)


def ri_fun(x, a, b):
    return (a-(b*(x)))


def exp_fun(x, a, b, c):
 #return a*np.exp(b*(x)) + c*np.exp(d*(x))
    return (a*np.exp(b*(x)))

def fun(x, a, b, c):
    return a*(x**(c))+b

def log_fun(x, a, b):
    return a*np.exp(b*(x))   #   a*np.exp(b*(x)) + c*np.log(d*(x))

popt, pcov = curve_fit(ri_fun, x44, y44, p0=[0, 0])

# popt2, pcov2 = curve_fit(log_fun, x33, y33, p0=[0, 0] )

print(popt)


# print("0.11A : ", fun(0.11,popt[0],popt[1]))
# print("0.22A : ", fun(0.22,popt[0],popt[1]))
# print("0.55A : ", fun(0.55,popt[0],popt[1]))
# print("1.1A : ", fun(1.1,popt[0],popt[1]))
# print("1.65A : ", fun(1.65,popt[0],popt[1]))
# print("2.2A : ", fun(2.2,popt[0],popt[1]))

y= []
for i in range(len(x44)):
    y.append(ri_fun(x44[i],popt[0],popt[1]))


plt.figure()
# plt.plot(exp_fun(x_idx, popt[0], popt[1], popt[2], popt[3]))
#plt.plot(log_fun(x_idx, popt2[0], popt2[1]))
plt.scatter(x44,y44)
#plt.scatter(x2,y2,color='red')

plt.plot(x44, y)

plt.show()