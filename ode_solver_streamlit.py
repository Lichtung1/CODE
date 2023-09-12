import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def ode_solver(equation_str, X_1, X_2, *initial_conditions):
    # Adjusting the ODE notation for sympy parsing
    equation_str = equation_str.replace("y''(x)", "Derivative(y(x),x,x)")
    equation_str = equation_str.replace("y'(x)", "Derivative(y(x),x)")

    # Setting up the symbols
    y, x, C1, C2 = sp.symbols('y x C1 C2')

    # Parsing the equation
    lhs, rhs = equation_str.split('=')
    eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))

    # Solve the ODE
    sol = sp.dsolve(eq)

    x_vals = np.linspace(X_1, X_2, 100)
    C1_vals = [-2, -1, 0, 1, 2, 3]
    fig, ax = plt.subplots()

    general_solution_text_str = str(sol)
    particular_solution_text_str = None

    if "C1" in str(sol) and "C2" not in str(sol):  # 1st order
        particular_solution = sol.subs(C1, sp.sympify(initial_conditions[0]))

        for val in C1_vals:
            general_solution = sol.subs(C1, val)
            label = f'C = {val}'
            ax.plot(x_vals, [general_solution.rhs.subs(x, val_x) for val_x in x_vals], 'b-')
            ax.annotate(label, xy=(x_vals[-1], general_solution.rhs.subs(x, x_vals[0])), fontsize=10)
            general_solution_text_str += f"\n{label}: {general_solution.rhs}"

        label = 'Particular Solution'
        ax.plot(x_vals, [particular_solution.rhs.subs(x, val_x) for val_x in x_vals], 'r-', label = 'Particular Solution')
        particular_solution_text_str = f"{label}: {particular_solution.rhs}"    
        ax.legend()
        
        ax.set_title("1st Order ODE Solutions")

    elif "C1" in str(sol) and "C2" in str(sol):  # 2nd order
        particular_solution = sol.subs({C1: sp.sympify(initial_conditions[0]), C2: sp.sympify(initial_conditions[1])})

        if not "General" in str(sol):
            label = 'Particular Solution'
            ax.plot(x_vals, [particular_solution.rhs.subs(x, val_x) for val_x in x_vals], 'r-', label = 'Particular Solution')
            particular_solution_text_str = f"{label}: {particular_solution.rhs}"    
            ax.legend()

        ax.set_title("2nd Order ODE Solutions")

    ax.set_xlabel('x')
    ax.set_ylabel('y(x)')
    ax.grid(True)
    ax.legend()
    
    st.write(f"General solution: {sol}")
    if particular_solution_text_str:
        st.write(f"Particular solution: {particular_solution_text_str}")

    return fig

# Streamlit UI
st.title("ODE Solver")

equation = st.text_input("Equation:", "y'(x)=x/4")
range_str = st.text_input("Range (e.g., -10,10):", "-10,10")
X_1, X_2 = map(float, range_str.split(','))
conditions = st.text_input("Initial Conditions (e.g. comma separated 1,2 for y(0)=1, y'(0)=2): ", "1").split(',')

if st.button('Solve'):
    try:
        if len(conditions) == 1:
            fig = ode_solver(equation, X_1, X_2, conditions[0])
        else:
            fig = ode_solver(equation, X_1, X_2, conditions[0], conditions[1])
        st.pyplot(fig)
    except ValueError:
        st.error("Please ensure inputs are correctly formatted.")
    except Exception as e:
        st.error(str(e))
