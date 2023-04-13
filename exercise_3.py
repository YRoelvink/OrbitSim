import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from constants import *
from exercise_2 import solve_twobody


# FUNCTION DEFINITIONS

def Zrotation(x, phi):
    # Function to rotate a vector x around the z-axis with angle phi
    rotation_rad = np.deg2rad(phi)
    rotation_axis = np.array([0, 0, 1])
    rotation_vector = rotation_rad * rotation_axis
    rotation = Rotation.from_rotvec(rotation_vector)
    return rotation.apply(x)


def solve_chaser(N, Theta, a_target):
    # Function used to compute the orbital parameters of the chaser's orbit
    a = a_target * (1 - np.deg2rad(Theta) / (2 * np.pi * N)) ** (2 / 3)
    e = a_target / a - 1
    return [a, e]


def ComputeDeltaV(N, Delta_Theta, pos_target, v_target, a_target, t_int, Reltol=1e-12, Abstol=1e-12):
    # Function that computes the Delta V needed to circularize the chaser's orbit at the point of minimum distance

    # Compute chaser parameters
    a, e = solve_chaser(N, Delta_Theta, a_target)

    # Compute apogee position and velocity
    ra = a_target
    va = np.sqrt(mu_e * ((2 / ra) - 1 / a))

    # Define initial position and velocity vectors
    r0 = np.array([ra, 0, 0])
    v0 = np.array([0, va, 0])

    # Define initial value problem & solve the two body problem
    X0 = np.concatenate((r0, v0), axis=None)
    x, y, z, r, vx, vy, vz, v = solve_twobody(X0, t_int, Reltol, Abstol)

    # Compute the minimal distance between the ISS and the chaser
    distance = np.sqrt((pos_target[0] - x) ** 2 + (pos_target[1] - y) ** 2 + (pos_target[2] - z) ** 2)
    min_distance = min(distance)

    # Compute delta V at position of minimum distance
    min_index = np.where(distance == min_distance)
    DeltaV = v_target[min_index] - v[min_index]
    return DeltaV


# QUESTIONS
# 3A

print("QUESTION 3A")
# Set variables
Delta_Theta = 100  # Initial Angular Separation [degree]
alt_ISS = 404  # altitude of the ISS [km]
Nrev = 12  # number of orbits needed for the chaser to intercept in the ISS
step_size = 10  # Seconds

# Compute Parameters ISS
max_time = (Nrev / 12) * 86400  # Seconds (86400 = 1 day)
a_ISS = Re + alt_ISS  # semi-major axis of the ISS orbit [km]
n_ISS = np.sqrt(mu_e / a_ISS ** 3)  # Mean Motion
tau_ISS = 2 * np.pi / n_ISS  # Period of Elliptical Curve [s]
v_ISS = np.sqrt(mu_e / a_ISS)  # (Circular) Orbital velocity of the ISS [km/s]

print(f"The orbital period of the ISS = {tau_ISS} s")

# Compute Initial vector of the ISS
r0_ISS = Zrotation(np.array([a_ISS, 0, 0]), Delta_Theta)
v0_ISS = Zrotation(np.array([0, v_ISS, 0]), Delta_Theta)
X0_ISS = np.concatenate((r0_ISS, v0_ISS), axis=None)

# Solve ISS two-body problem
t_ISS = np.linspace(0, max_time, int(max_time / step_size) + 1)
sol_x_ISS, sol_y_ISS, sol_z_ISS, sol_r_ISS, sol_vx_ISS, sol_vy_ISS, sol_vz_ISS, sol_v_ISS = solve_twobody(X0_ISS, t_ISS)

# Plot ISS Orbit
fig = plt.figure(figsize=(7, 7))
ax = fig.gca()
ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.scatter(sol_x_ISS, sol_y_ISS, color='k', s=2, label="ISS")

# Plot Earth as a blue circle and radius Re
Earth = plt.Circle((0, 0), Re, color='blue')
ax.add_patch(Earth)

print("\n")
print("QUESTION 3B")


# Compute chaser's orbital parameters
a_chaser, e_chaser = solve_chaser(Nrev, Delta_Theta, a_ISS)
print(f"The semi-major axis and the eccentricity of the chaser are {a_chaser:.2f} km and {e_chaser:.2f}")

# Set parameters for the Chaser
ra_chaser = a_ISS   # Apoapsis radius of the chaser is equal to that of the ISS
r0_chaser = np.array([ra_chaser, 0, 0])
n_chaser = np.sqrt(mu_e / a_chaser ** 3)  # Mean Motion
tau_chaser = 2 * np.pi / n_chaser  # Period of Elliptical Curve [s]
va_chaser = np.sqrt(mu_e * ((2 / ra_chaser) - 1 / a_chaser))  # Orbital Apoapsis velocity of the chaser
v0_chaser = np.array([0, va_chaser, 0])

# Set initial position and velocity vector of the chaser
X0_chaser = np.concatenate((r0_chaser, v0_chaser), axis=None)

# Solve Chaser two-body problem
sol_x_chaser, sol_y_chaser, sol_z_chaser, sol_r_chaser, sol_vx_chaser, sol_vy_chaser, sol_vz_chaser, sol_v_chaser = solve_twobody(
    X0_chaser, t_ISS)

# Add the Chaser's orbit to the previous plot
ax.scatter(sol_x_chaser, sol_y_chaser, color='r', s=2, label="Chaser")
plt.legend()
plt.show()

# Compute the distance between the ISS and the chaser
distance = np.sqrt((sol_x_ISS - sol_x_chaser) ** 2 + (sol_y_ISS - sol_y_chaser) ** 2 + (sol_z_ISS - sol_z_chaser) ** 2)

# Plot the distance between the ISS and the chaser
fig2 = plt.figure()
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.plot(t_ISS / 3600, distance)
plt.xlabel("Time (hours)")
plt.ylabel("Distance (km)")
plt.xlim(0, max_time / 3600)
plt.xticks(np.arange(0, max_time / 3600 + 3, step=3))
plt.ylim(0)
plt.show()

# Compute the minimum distance obtained
min_distance = min(distance)
print(f'The minimum distance between the chaser and the ISS = {min_distance * 1000:.2f} m')
print("\n")

# %% 3c
print("QUESTION 3C")
# Set limits of revolutions needed for the chaser to cath up
Nrev_min = 2
Nrev_max = 30
step_size = 10  # Seconds

# Define parameter arrays
max_time = (Nrev_max / 12) * 86400  # Seconds (86400 = 1 day)
t_deltaV = np.linspace(0, max_time, int(max_time / step_size) + 1)
Nrev = np.linspace(Nrev_min, Nrev_max, Nrev_max - Nrev_min + 1)

# Initialize solution array
DeltaV = np.empty(np.size(Nrev))

# Solve two body problem for the ISS velocity & position array
sol_x_ISS, sol_y_ISS, sol_z_ISS, _, _, _, _, sol_v_ISS = solve_twobody(X0_ISS, t_deltaV)
pos_vec_SS = [sol_x_ISS, sol_y_ISS, sol_z_ISS]
v_vec_ISS = sol_v_ISS

# Compute delta V required for circulisation for a given number of catch-up orbits
for i, N in enumerate(Nrev):
    DeltaV[i] = ComputeDeltaV(N, Delta_Theta, pos_vec_SS, v_vec_ISS, a_ISS, t_deltaV)

# Plots solution
fig = plt.figure()
plt.plot(Nrev, DeltaV)
plt.xlabel("$N_{rev}$")
plt.ylabel("$\Delta$v (km/s)")
plt.xlim(0, Nrev_max)
plt.ylim(0)
plt.show()

print("\n")

# %% 3d
print("QUESTION 3D")

# Define variables
min_perigee = 60 + Re  # km
max_perigee = 210 + Re  # km
step_size = 1  # km

# Define initial parameters of the chaser
v_init = v_ISS
ra_deorbit = a_ISS

# Define range of perigee radii
perigee = np.linspace(min_perigee, max_perigee, int((max_perigee - min_perigee) / step_size) + 1)

# Initialise solution array
DeltaV_deorbit = np.empty(np.size(perigee))

# Compute Delta V for a range of perigee radii
for i, p in enumerate(perigee):
    a_deorbit = (p + ra_deorbit) / 2
    v_deorbit = np.sqrt(mu_e * (2 / ra_deorbit - 1 / a_deorbit))
    DeltaV_deorbit[i] = v_deorbit - v_init

# Plot the delta V for against the corresponding perigee altitude
fig = plt.figure()
plt.plot(perigee - Re, DeltaV_deorbit)
plt.xlabel("Perigee altitude (km)")
plt.ylabel("$\Delta$V (km/s)")
plt.xlim(min_perigee - Re, max_perigee - Re)
plt.xticks(np.arange(min_perigee - Re, max_perigee + 15 - Re, step=15))
plt.show()