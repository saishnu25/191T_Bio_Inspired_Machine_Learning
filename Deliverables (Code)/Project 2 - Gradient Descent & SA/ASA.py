import numpy as np
import sympy as sp



def objective_function(objective_func, position):
    return objective_func(*position)
#end of function



def logarithmic_cooling(initial_temperature, k, current_time):
    """
    Calculate the temperature at the current time using a logarithmic cooling schedule.

    Parameters:
    - initial_temperature: The initial temperature (T0).
    - k: A constant controlling the cooling rate.
    - current_time: The current iteration or time step (t).

    Returns:
    - The temperature at the current time.
    """
    return initial_temperature / (1 + k * np.log(current_time + 1))
#end of function

def adaptive_simulated_annealing(equation, initial_solution, max_iterations=1000, T0=100.0, kappa=0.01):
    #initialize
    current_solution = initial_solution
    best_solution = current_solution
    best_energy = objective_function(equation, current_solution)
    step_size = 0.1 #initial step size

    #Just to count iterations
    count=0
    for t in range(max_iterations):
        #adjust temp
        temperature = logarithmic_cooling(T0, kappa, t)
        #find new candidiates
        candidate_x = current_solution[0] + np.random.normal(scale=step_size)
        candidate_y = current_solution[1] + np.random.normal(scale=step_size)
        candidate_solution = (candidate_x, candidate_y)
        candidate_energy = objective_function(equation, candidate_solution)

        #check energy to see if it is going in the right direction
        if candidate_energy < best_energy:
            best_solution = candidate_solution
            best_energy = candidate_energy

        #Boltzmann probability/metropolis crietrion
        delta_energy = candidate_energy - best_energy
        acceptance_prob = np.exp(-delta_energy / temperature)
        if np.random.rand() < acceptance_prob:
            current_solution = candidate_solution

        count = count + 1
    #end of loop
    return best_solution, best_energy, count
#end of function

#I can make a def numpy function for the equation but I think this looks visually nicer
x, y = sp.symbols('x y')
ras_fxy = 20 + (x**2) + (y**2) - 10 * (sp.cos(2 * sp.pi * x) + sp.cos(2 * sp.pi * y))
objective_func = sp.lambdify((x, y), ras_fxy, 'numpy')  # Create a lambdified function once

initial_solution = [-1.5,2]
intial_conditions = [(2,-2), (1.5,-1), (8,3), (0,0), (0.1,0.1), (0.5,0.7)]

initial_temperature = 100.0  # Initial temperature
cooling_constant = 10.0  # Cooling rate constant for the logarithmic cooling schedule
max_iterations = 100000  # Maximum number of iterations

for i in intial_conditions: 
    best_position, best_energy,count = adaptive_simulated_annealing(objective_func,i, max_iterations, initial_temperature, kappa= 0.01)
    print(f"Best position found: {best_position}, with energy: {best_energy}\n")
    print(f"Total Iteration:{count}\n NEW CYCLE\n" )
#end of loop