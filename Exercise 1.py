import matplotlib.pyplot as plt
import numpy as np

from constants import *


# FUNCTION DEFINITIONS

def Kepler(E, M, e):
    # Function used to define the Kepler Equation
    return E - e * np.sin(E) - M


def KeplerPrime(E, e):
    # Function used to define the first derivative of the Kepler Equation
    return 1 - e * np.cos(E)


def SolveKepler(M, e, a, tol, printing=True):
    # Use M as first guess for E
    Ei = M

    # Iterate over Newton-Raphson Method & update the Eccentric anomaly
    # untill the pre-defined tolerance is met 
    i = 0
    while np.abs(Kepler(Ei, M, e) / KeplerPrime(Ei, e)) > tol:
        Er = Ei - Kepler(Ei, M, e) / KeplerPrime(Ei, e)
        if printing:
            print(f'After {i + 1} iteration E = {Er:.5f} rad and the error = {Kepler(Er, M, e)}')
        Ei = Er
        i += 1

    if printing:
        print(f'The number of iterations needed to converge = {i} and the final value of E = {Ei:.5f} rad')

    # Compute true anomaly and the radius from the acquired Eccentric Anomaly
    theta = np.arccos((np.cos(Ei) - e) / (1 - e * np.cos(Ei)))
    r = a * (1 - e * np.cos(Ei))
    return [Ei, theta, r]


def comp_cartesian(E, a, b):
    # Function that computes the Cartesian coordinates from the orbital parameters
    x = a * (np.cos(E) - e)
    y = b * np.sin(E)
    return (x, y)


def Calc_Delta_T(E1, E2, e, n):
    # Function to compute the time difference between two orbital points
    M1 = E1 - e * np.sin(E1)
    M2 = E2 - e * np.sin(E2)
    dT = np.abs(M2 - M1) / n
    return dT


# QUESTIONS
# 1B
print("QUESTION 1B")

# Define Variables
e = 0.25  # Eccentricity
a = 24000  # Semi-major axis [km]
M_deg = 180  # Mean anomaly [degrees]
tol = 1e-12  # Solver's error tollerance

# Solve the Kepler equation
E, theta, r = SolveKepler(np.deg2rad(M_deg), e, a, tol)

print(f"For M = {M_deg} degree and e = {e}, theta & r are: {theta:.2f} rad and {r:.2f} km")
print("\n")


# 1C
print("QUESTION 1C")
# Define Variables
e = 0.72  # Eccentricity
a = 24000  # Semi-major axis [km]
tol = 1e-12  # Solver's error tolerance
step_size = 15  # Seconds
max_time = 86400  # Seconds (86400 s = 1 day)

# Compute Parameters
n = np.sqrt(mu_e / a ** 3)  # Mean Motion
tau = 2 * np.pi / n  # Period of Elliptical Curve

# Define time array with a step_size of 15 seconds for a single day
t = np.linspace(0, max_time, int(max_time / step_size) + 1)

# Compute the mean anomaly from the time array
M_MEO = n * t

# Compute the index after which the orbit will be repeated
max_orbit_index = int(tau / step_size)

# Initialize solution arrays (radius, eccentric anomaly and altitude)
r_MEO = np.empty(np.size(M_MEO))
E_MEO = np.copy(r_MEO)
h_MEO = np.copy(r_MEO)

# Compute altitudes from the eccentric anomalies
for i, M in enumerate(M_MEO):
    E_MEO[i], _, r_MEO[i] = SolveKepler(M, e, a, tol, False)
    h_MEO[i] = r_MEO[i] - Re

# Plotting the altitude for a single orbit    
fig, (ax1, ax2) = plt.subplots(1, 2, sharey="True", figsize=(10, 5))
ax1.plot(t[:max_orbit_index] / tau * 100, h_MEO[:max_orbit_index])
ax1.set_xlabel("Time (% of orbital period)")
ax1.set_xlim(0, 100)
ax1.set_ylabel("Altitude (km)")
ax1.set_ylim(0)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Plotting the altitude for a single day
ax2.plot(t / 3600, h_MEO)
ax2.set_xlabel("Time (hours)")
ax2.set_xticks(np.arange(0, 27, step=3))
ax2.set_xlim(0, 24)
plt.show()
print("\n")


# 1D
print("QUESTION 1D")
# Define variables
b = a * np.sqrt(1 - e ** 2)  # Semi-minor axis [km]

# Initialize arrays
x_MEO = np.copy(r_MEO)
y_MEO = np.copy(r_MEO)

# Compute cartesian coordinates of the satellite
for i, E in enumerate(E_MEO):
    x_MEO[i], y_MEO[i] = comp_cartesian(E, a, b)

# Compute variables used in Eclipse equations
p = a * (1 - e ** 2)
alpha = (Re * e) ** 2 + p ** 2
beta = 2 * (Re ** 2) * e
gamma = (Re ** 2) - (p ** 2)

# Solve Eclipse equations for true anomalies & associated eccentric anomalies
Delta = np.sqrt(beta ** 2 - 4 * alpha * gamma)
theta_eclipse_1 = np.arccos((-beta + Delta) / (2 * alpha))
theta_eclipse_2 = np.arccos((-beta - Delta) / (2 * alpha))
E_eclipse_1 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(theta_eclipse_1 / 2))
E_eclipse_2 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(theta_eclipse_2 / 2))

# Compute cartesian coordinates of the eclipse points
x_eclipse_1, y_eclipse_1 = comp_cartesian(E_eclipse_1, a, b)
x_eclipse_2, y_eclipse_2 = comp_cartesian(E_eclipse_2, a, b)

# Plot Earth as a blue circle and raius Re
Earth = plt.Circle((0, 0), Re, color='blue')
fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
ax.add_patch(Earth)
ax.plot(x_MEO, y_MEO, color='k')
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")

# Plot Eclipse points
ax.scatter(x_eclipse_1, y_eclipse_1, marker="*", color="r")
ax.scatter(x_eclipse_1, -y_eclipse_1, marker="*", color="r")
ax.scatter(x_eclipse_2, y_eclipse_2, marker="o", color="r")
ax.scatter(x_eclipse_2, -y_eclipse_2, marker="o", color="r")

# Compute Eclipse times and plot output
Eclipse_time_1 = Calc_Delta_T(-E_eclipse_1, E_eclipse_1, e, n)
print(f"Duration stayed in Eclipse 1 = {(Eclipse_time_1 / 60):.2f} min")
Eclipse_time_2 = tau - Calc_Delta_T(-E_eclipse_2, E_eclipse_2, e, n)
print(f"Duration stayed in Eclipse 2 = {(Eclipse_time_2 / 60):.2f} min")
