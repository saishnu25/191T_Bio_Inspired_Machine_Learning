#This is the source file for the Gradient Descent Problem
#will be utilizing sympy for the gradient calculations
import sympy as sp
import numpy as np


"""

Note that Before you use variables in an expression, we need to define them as SymPy symbols
This distinguishes them from regular Python variables and allows SymPy to perform symbolic computation with them.

"""
def general_gradient_compute(func_exp, vars):
    """
    Compute the gradient/derivative of a function in respect to a variable
    
    func_exp will be a sympy expression representing the function that will be computed
    Vars will be a list of sympy variables that will be used for each partial derivative
    Return: a gradient of the function with respect to each variable
    """
    if len(vars) > 1:
        grad = [sp.diff(func_exp, var) for var in vars]

    else: 
        #just a simple derivative
        grad = [sp.diff(func_exp, vars[0])]

    return grad
#end of function


#Cosine Learning rate from Classroom slides
def cosine_learning_rate(alpha_init, t , T):
    return alpha_init * (1 + np.cos((np.pi * t)/T)) / 2
#end of function


#apply gradient descent with momentum
def gradient_descent(func_exp, vars, init_params, alpha_init, gamma, max_iter, eps = 1e-6):
    """
    func_exp: sympy expression of the function to minimize work
    vars: list of sympy variable in function
    init_params: initial parameters as a numpy array
    alpha_init: intial learning rate
    gamma: Momentum coefficient
    max_iter: max iterations
    esp: convergence threshold
    return: parameters of minimum

    """
    #make intial parameters into a numpy array of float type
    params = np.array(init_params, dtype=float)

    m = np.zeros(len(params))

    #find gradient expressions
    grad_expressions = general_gradient_compute(func_exp, vars)

    
    

    for t in range(max_iter):
        #hold gradient vals and reset 
        computed_gradients = []
        #evaluate gradient vals at current parameters
        for grad_expr in grad_expressions:
            #create a dictionary that maps each variable to its current value
            subs_dict = dict(zip(vars,params))

            #eval current gradient expression
            gradient_val = grad_expr.evalf(subs = subs_dict)

            # Append the computed gradient value to our list of gradients
            computed_gradients.append(gradient_val)
        
        #convert list into numpy array
        gradients = np.array(computed_gradients, dtype=float)

        alpha_t = cosine_learning_rate(alpha_init, t, max_iter)

        #update momentum terms
        m = gamma * m + alpha_t * gradients

        #update the parameters
        params -= m

        # Check for convergence
        if np.linalg.norm(m) < eps:
            print(f"Converged after {t+1} iterations.")
            break
    #end of loop
    return params
        #end of loop
#end of function

#Test general gradient function
#Two variable function f(x, y) = x^2 + y^2
#x = sp.symbols('x')
#y = sp.symbols('y')
#z = sp.symbols('z')
#f_xyz = x**2 + y**2 + z**2
#grad_multi = general_gradient_compute(f_xyz, [x, y, z])
#print("Gradient of f(x, y) = x^2 + y^2:", grad_multi)
"""
test with simple function
the function will have three minimums
I have no ideas where the other minimums are located
The absolute minimum is located at point (3/2,3/2,-27/8)
the function is f(x,y) = x^4 - 2x^3 + y^4 - 2y^3

"""

x = sp.symbols('x')
y = sp.symbols('y')
f_xy = (x**4) - (2 * x**3) + (y**4) - (2* y**3)

alpha_init = 0.01
gamma = 0.8
max = 100

final_params = gradient_descent(f_xy,[x,y], [3/5,3/5],alpha_init,gamma,max)

print("absolute min is located at", final_params)

#Testing with Rastrigin function

ras_fxy = 20 + (x**2) + (y**2) - 10 * (sp.cos(2 * sp.pi * x) + sp.cos(2 * sp.pi * y))

max = 100000
final_params = gradient_descent(ras_fxy,[x,y], [0.5,0.7],alpha_init,gamma,max)

print("absolute min for Ras(x,y) is located at", final_params)
position = ras_fxy.subs({x: final_params[0], y: final_params[1]}).evalf()
print("\n The position: ", position)