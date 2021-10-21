import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from constants import *


# FUNCTION DEFINITIONS

def twobody(t,X):
    x = X[0]
    y = X[1]
    z = X[2]
    r = np.sqrt(x**2 + y**2 + z**2)

    xdot = X[3]
    ydot = X[4]
    zdot = X[5]

    return [xdot, ydot, zdot, -mu_e/(r**3)*x, -mu_e/(r**3)*y, -mu_e/(r**3)*z]

def solve_twobody(X0, t, Reltol = 1e-12, Abstol = 1e-12):

    # Solve ODE
    sol = solve_ivp(twobody, (0,max(t)), X0, t_eval = t, rtol = Reltol, atol = Abstol)

    # Compute Positional solutions
    sol_x = sol.y[0,:]
    sol_y = sol.y[1,:]
    sol_z = sol.y[2,:]
    sol_r = np.sqrt(sol_x**2 + sol_y**2 + sol_z**2)

    # Compute Velocity solutions
    sol_vx = sol.y[3,:]
    sol_vy = sol.y[4,:]
    sol_vz = sol.y[5,:]
    sol_v = np.sqrt(sol_vx**2 + sol_vy**2 + sol_vz**2)

    return [sol_x, sol_y, sol_z, sol_r, sol_vx, sol_vy, sol_vz, sol_v]


# QUESTIONS
# 2C

print("QUESTION 2C")
# define variables
r0 = [7115.804, 3391.696, 3492.221]         # km
v0 = [-3.762, 4.063, 4.184]                 # km/s
Reltol = 1e-12                              # Solver's Relative error tolerance
Abstol = 1e-12                              # Solver's Absolute error tolerance
stepsize = 10                               # Seconds
max_time = 86400                            # Seconds (86400 = 1 day)
plt_resolution = 200                        # 3D plotting resolution

# Initialize time and initial position vector
t = np.linspace(0, max_time, int(max_time / stepsize)+1)
X0 = np.concatenate((r0, v0), axis = None)

# Solve ODE
sol_x, sol_y, sol_z, sol_r, sol_vx, sol_vy, sol_vz, sol_v = solve_twobody(X0, t, Reltol, Abstol)

# Plot position magnitude over time
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='True')
ax1.plot(t/3600, sol_r)
ax1.set_ylabel("Radius (km)")
ax1.set_xlim(0,24)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
ax1.set_xticks(np.arange(0,27,step = 3))

# Plot the velocity magnitude over time
ax2.plot(t/3600, sol_v)
ax2.set_ylabel("Velocity (km/s)")
ax2.set_xlabel("Time (hours)")
plt.show()

# Plot the orbit in a 3D plane
fig2 = plt.figure(figsize = (10,7))
ax3 = fig2.add_subplot(111, projection='3d')
ax3.ticklabel_format(axis = 'both', style = 'sci', scilimits = (0,0))
ax3.set_xlabel("x (km)")
ax3.set_ylabel("y (km)")
ax3.set_zlabel("z (km)")
ax3.scatter(sol_x, sol_y, zs = sol_z, color = 'k', s=2)

# Add the Earth to the plot
theta_Earth = np.linspace(0,2*np.pi,plt_resolution)
phi_Earth = np.linspace(0, np.pi, plt_resolution)
x_Earth = Re*np.outer(np.cos(theta_Earth), np.sin(phi_Earth))
y_Earth = Re*np.outer(np.sin(theta_Earth), np.sin(phi_Earth))
z_Earth = Re*np.outer(np.ones(np.size(theta_Earth)), np.cos(phi_Earth))
ax3.plot_surface(x_Earth, y_Earth, z_Earth)
plt.show()
print("\n")


# 2D
print("QUESTION 2D")

# Compute Specific Energies
Ek = sol_v**2 / 2           # Specific Kinetic Energy [km^2 / s^2]
Ep = -mu_e/sol_r            # Specific Potential Energy [km^2 / s^2]
Etot = Ek + Ep              # Total Specific Energy [km^2 / s^2]

# Compute Specific Angular Momentum
r_vec = [sol_x, sol_y, sol_z]
v_vec = [sol_vx, sol_vy, sol_vz]
h = np.linalg.norm(np.cross(r_vec, v_vec, axis=0), axis=0)

# Plot Specific Energies
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='True', figsize=(8, 5))
ax1.plot(t/3600, Ek, label= 'Specific Kinetic Energy')
ax1.plot(t/3600, Ep, label = 'Specific Potential Energy')
ax1.plot(t/3600, Etot, label = 'Total Specific Energy')
ax1.legend()
ax1.set_ylabel("E ($km^2/s^2$)")

ax2.plot(t/3600, h)
ax2.ticklabel_format(axis = 'y', style ='sci', scilimits=(0, 0))
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("h ($km^2/s$)")
ax2.set_xlim(0,24)
ax2.set_xticks(np.arange(0,27,step = 3))
ax2.set_ylim(max(h)*0.99, max(h)*1.01)
plt.show()
print("\n")


# 2E
print("QUESTION 2E")
# define variables
r0 = [0, 0, 8550]                           # km
v0 = [0, -7.0, 0]                           # km/s
Reltol = 1e-12                              # Solver's Relative error tolerance
Abstol = 1e-12                              # Solver's Absolute error tolerance
stepsize = 10                               # Seconds
max_time = 86400                            # Seconds (86400 = 1 day)
plt_resolution = 200                        # 3D plotting resolution

# Compute parameters
t = np.linspace(0, max_time, int(max_time / stepsize)+1)
X0 = np.concatenate((r0, v0), axis = None)

sol_x, sol_y, sol_z, sol_r, sol_vx, sol_vy, sol_vz, sol_v = solve_twobody(X0, t, Reltol, Abstol)

# Plot position magnitude over time
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
ax1.plot(t/3600, sol_r)
ax1.set_ylabel("Radius (km)")
ax1.set_xlim(0,24)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
ax1.set_xticks(np.arange(0,27,step = 3))

# Plot the velocity magnitude over time
ax2.plot(t/3600, sol_v)
ax2.set_ylabel("Velocity (km/s)")
ax2.set_xlabel("Time (hours)")
plt.show()

# Plot the orbit in a 3D plane
fig2 = plt.figure(figsize = (10,7))
ax3 = fig2.add_subplot(111, projection='3d')
ax3.ticklabel_format(axis = 'both', style = 'sci', scilimits = (0,0))
ax3.set_xlabel("x (km)")
ax3.set_ylabel("y (km)")
ax3.set_zlabel("z (km)")
ax3.set_xlim(-max(sol_r), max(sol_r))
ax3.set_ylim(-max(sol_r), max(sol_r))
ax3.set_zlim(-max(sol_r), max(sol_r))
ax3.scatter(sol_x, sol_y, zs = sol_z, color = 'k', s=2)

# Add the Earth to the plot (using the definition from before)
ax3.plot_surface(x_Earth, y_Earth, z_Earth)
plt.show()

# Compute Orbital Parameters
r_a = max(sol_r)            # Periapsis radius [km]
r_p = min(sol_r)            # Apoapsis radius [km]
a = (r_p + r_a)/2           # semi-major axis [km]
print(f"The semi-major axis of the satellite = {a:.2f} km")

e = r_a/a - 1               # Eccentricity
print(f"The eccentricity of the satellite = {e:.2f}")

n = np.sqrt(mu_e/a**3)      # Mean Motion
tau = 2*np.pi / n           # Period of Elliptical Curve [s]
print(f"The period of the satellite = {tau:.2f} s")

h_vec = np.cross(r0, v0)
normalized_h_vec = h_vec / np.sqrt(np.sum(h_vec**2))
z_axis = [0,0,1]
i = np.arccos(np.dot(normalized_h_vec, z_axis))
print(f"The inclination of the satellite = {np.rad2deg(i)} degrees")