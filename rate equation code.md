# rate-equation
simulation code for rate equation

Run the simulation in Python3 with needed configuration. Parameters in the code are given as example. 

# concentration dependency
import numpy as np
import matplotlib.pyplot as plt

# Constants
kB = 8.6173e-5  # Boltzmann constant in eV/K

# Temperature-dependent rate equation model
def temperature_dependent_lifetime(T, A_YbEr_base, A_Er, F, Delta_E_YbEr, g_Er, Yb_percentage, total_ions_base):
    # Adjust A_YbEr based on Yb3+ concentration (for illustration purposes)
    A_YbEr = A_YbEr_base / (1 + 0.1 * Yb_percentage)

    # Calculate Yb3+ concentration
    C_Yb = Yb_percentage * total_ions_base

    # Temperature-dependent rates
    w_YbEr = A_YbEr * np.exp(-Delta_E_YbEr / (kB * T))
    
    # Larger difference in thermalization for illustration
    b2 = (g_Er / (1 + 5 * Yb_percentage)) * np.exp(2 * Delta_E_YbEr / (kB * T))
    
    # Reciprocal lifetime (non-radiative process)
    inv_t_YbEr = 1 / (2 * F) * ((1 / (1 - b2**2)) * A_YbEr * A_Er - (1 / 4) * b2 / (1 - b2**2) * A_YbEr**2)
    
    return 1 / inv_t_YbEr

# Simulation parameters
T_values = np.linspace(300, 500, 100)  # Temperature range in Kelvin

# Constants and parameters (adjust based on your specific case)
A_YbEr_base = 1e6  # Base spontaneous transition rate Yb to Er in s^-1
A_Er = 1e6         # Spontaneous transition rate Er emission in s^-1
F = 1              # Filling factor
Delta_E_YbEr = 0.5 # Energy separation between Yb and Er levels in eV
g_Er = 2           # Degeneracy of Er excited state
total_ions_base = 1e20  # Base total rare-earth ion concentration in cm^-3

# Yb3+ doping concentrations as a percentage of total ions
Yb_doping_percentages = [0, 5, 10, 15, 20]

# Save the results in text files and plot
plt.figure(figsize=(10, 6))

for Yb_percentage in Yb_doping_percentages:
    # Simulate temperature-dependent lifetime decay
    lifetime_model = np.vectorize(temperature_dependent_lifetime)
    lifetime_T = lifetime_model(T_values, A_YbEr_base, A_Er, F, Delta_E_YbEr, g_Er, Yb_percentage / 100, total_ions_base)

    # Calculate the average lifetime
    avg_lifetime = np.mean(lifetime_T)

    # Save the data to a text file
    data = np.column_stack((T_values, lifetime_T))
    filename = f'lifetime_data_YbDoping_{Yb_percentage}.txt'
    np.savetxt(filename, data, header=f'Temperature (K)  Lifetime (s)\nAverage Lifetime: {avg_lifetime:.6e} s', comments='')

    # Plot the results
    plt.plot(T_values, lifetime_T, label=f'Yb3+ Doping: {Yb_percentage}% of Total Ions', linewidth=2)

plt.xlabel('Temperature (K)')
plt.ylabel('Lifetime (s)')
plt.title('Temperature-Dependent Lifetime Decay for Different Yb3+ Doping Concentrations')
plt.legend()
plt.grid(True)
plt.show()


# pumping powder dependency
import numpy as np
import matplotlib.pyplot as plt

# Constants
kB = 8.6173e-5  # Boltzmann constant in eV/K

# Temperature-dependent rate equation model with Pump Power and Saturation
def temperature_dependent_lifetime(T, A_YbEr_base, A_Er, F, Delta_E_YbEr, g_Er, Yb_percentage, total_ions_base, P):
    # Adjust A_YbEr based on Yb3+ concentration and Pump Power
    A_YbEr = A_YbEr_base / (1 + 0.1 * Yb_percentage) * (1 + 0.2 * P)

    # Calculate Yb3+ concentration
    C_Yb = Yb_percentage * total_ions_base

    # Temperature-dependent rates
    w_YbEr = A_YbEr * np.exp(-Delta_E_YbEr / (kB * T))
    
    # Larger difference in thermalization for illustration
    b2 = (g_Er / (1 + 5 * Yb_percentage)) * np.exp(2 * Delta_E_YbEr / (kB * T))
    
    # Reciprocal lifetime (non-radiative process)
    inv_t_YbEr = 1 / (2 * F) * ((1 / (1 - b2**2)) * A_YbEr * A_Er - (1 / 4) * b2 / (1 - b2**2) * A_YbEr**2)
    
    return 1 / inv_t_YbEr

# Simulation parameters
T_values = np.linspace(300, 500, 100)  # Temperature range in Kelvin
P_values = [1, 5, 10, 20, 50, 100, 200, 500]  # Pump Power values

# Constants and parameters (adjust based on your specific case)
A_YbEr_base = 1e6  # Base spontaneous transition rate Yb to Er in s^-1
A_Er = 1e6         # Spontaneous transition rate Er emission in s^-1
F = 1              # Filling factor
Delta_E_YbEr = 0.5 # Energy separation between Yb and Er levels in eV
g_Er = 2           # Degeneracy of Er excited state
total_ions_base = 1e20  # Base total rare-earth ion concentration in cm^-3
Yb_percentage = 10  # Yb3+ doping concentration as a percentage of total ions

# Save the results to a text file
filename = 'simulation_results.txt'

with open(filename, 'w') as file:
    file.write('Temperature(K)\t')
    for P in P_values:
        file.write(f'Pump Power {P} W\t')
    file.write('\n')

    for T in T_values:
        file.write(f'{T}\t')
        for P in P_values:
            # Simulate temperature-dependent lifetime decay
            lifetime_model = np.vectorize(temperature_dependent_lifetime)
            lifetime_T = lifetime_model(T, A_YbEr_base, A_Er, F, Delta_E_YbEr, g_Er, Yb_percentage / 100, total_ions_base, P)
            file.write(f'{lifetime_T}\t')
        file.write('\n')

# Plot the results for different Pump Powers
plt.figure(figsize=(10, 6))

for P in P_values:
    # Simulate temperature-dependent lifetime decay
    lifetime_model = np.vectorize(temperature_dependent_lifetime)
    lifetime_T = lifetime_model(T_values, A_YbEr_base, A_Er, F, Delta_E_YbEr, g_Er, Yb_percentage / 100, total_ions_base, P)

    # Plot the results
    plt.plot(T_values, lifetime_T, label=f'Pump Power: {P} W', linewidth=2)

plt.xlabel('Temperature (K)')
plt.ylabel('Lifetime (s)')
plt.title('Saturation Effect: Temperature-Dependent Lifetime Decay for Different Pump Powers')
plt.legend()
plt.grid(True)
plt.show()
