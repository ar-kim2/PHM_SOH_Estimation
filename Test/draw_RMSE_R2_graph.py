import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

x_list = [1,2,3,4,5,6,7,8,9,10,11]

discharge_RMSE_list = [0.02659133, 0.0077989195, 0.017338483, 0.015900642, 0.008053723, 0.01793397, 0.01593952, 0.007096790, 0.022676848, 0.008655923, 0.0073971413]
charge_RMSE_list = [0.060639277, 0.010651129, 0.020587474, 0.018080663, 0.005668315, 0.020851132, 0.019806914, 0.0076374705, 0.052111097, 0.013949345, 0.005961954]

discharge_R2_list = [0.99213964, 0.99855185, 0.99493706, 0.99364287, 0.9903974, 0.9936739, 0.9941038, 0.988441, 0.98707426, 0.9981967, 0.9914747]
charge_R2_list = [0.9556543, 0.99739426, 0.9908248, 0.9918313, 0.9928594, 0.9908935, 0.99251, 0.9842556, 0.9576704, 0.9960441, 0.98810476]


fig, ax1 = plt.subplots()
ax1.plot(x_list, discharge_R2_list, label='discharge', linestyle='--', marker='o')
ax1.plot(x_list, charge_R2_list, label='charge', linestyle='--', marker='o')
ax1.set_xlabel('Test Case')
ax1.set_ylabel('R2')
ax1.legend(loc='upper right')
ax1.set_ylim(0.88, 1.01)
ax1.set_xlim(0.0001, 11.999)
ax1.set_xticks(x_list)

ax2 = ax1.twinx()
ax2.plot(x_list, discharge_RMSE_list, label='discharge', color='green', linestyle='--', marker='o')
ax2.plot(x_list, charge_RMSE_list, label='charge', color='red', linestyle='--', marker='o')
ax2.set_ylabel('RMSE')
ax2.legend(loc='lower right')
ax2.set_ylim(0.0, 0.12)

plt.show()