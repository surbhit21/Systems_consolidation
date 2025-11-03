
import matplotlib.pyplot as plt
import numpy as np
alpha_0 = 0
target = 2
t_final = 30000
dt = 1
t = np.arange(0,t_final,dt, dtype=float)
alpha_i = np.empty_like(t, dtype=float)
alpha_i[0] = alpha_0
tau_alpha_p = 10
tau_alpha_m = 1000
goes_down = False
tol = 1e-5
t_up_time = 100
first_hit_time = t_final
for i in range(1,t_final//dt):
    if not goes_down:
        dalpha_dt = (2-alpha_i[i-1])/tau_alpha_p
    else:
        dalpha_dt = (1-alpha_i[i-1])/tau_alpha_m
    alpha_i[i] = alpha_i[i-1] + (dalpha_dt*dt)
    if abs(alpha_i[i] - target) <= tol and t[i] < first_hit_time:
        first_hit_idx = i
        first_hit_time = t[i]
    if t[i] > first_hit_time + t_up_time:
        goes_down = True
print(first_hit_time)
plt.plot(t/1e+3,alpha_i)
plt.show()

