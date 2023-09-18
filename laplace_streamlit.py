import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import streamlit as st

# Define the main function that computes and plots the Laplace Transform
def compute_and_plot(func_str, alpha_domain, omega_domain, F_s_domain, n, lines):
    t, s, a, w = sp.symbols('t s a w')
    func_sym = sp.sympify(func_str)
    
    F_s = sp.laplace_transform(func_sym, t, s)[0]
    F_s_substituted = F_s.subs(s, a + 1j*w)
    F_func = sp.lambdify((a, w), F_s_substituted, 'numpy')
    

    a_vals, w_vals = np.meshgrid(np.linspace(*alpha_domain, n), np.linspace(*omega_domain, n))
    F_magnitude_matrix = np.abs(F_func(a_vals, w_vals))
    tolerance = F_s_domain[1]*0.1
    F_magnitude_matrix[F_magnitude_matrix > F_s_domain[1]+tolerance] = F_s_domain[1]+tolerance
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if lines:
        edgecolor_set = 'k' 
    else:
        edgecolor_set = 'none' 
            
    surf = ax.plot_surface(a_vals, w_vals, F_magnitude_matrix, cmap='cool', edgecolor=edgecolor_set, linewidth=0.25)
    surf.set_clim(*F_s_domain)
    
    ax.set_title(f'Laplace Transform of f(t)={func_str}')
    ax.set_xlabel('α (Real)')
    ax.set_ylabel('ω (Imaginary)')
    ax.set_zlabel('|F(s)|')
    ax.set_zlim(*F_s_domain)

    return fig

# Define the Streamlit app
st.title("Laplace Transform Visualizer")
    
# UI Components
func_str = st.text_input("Function f(t)", "exp(-t)")
alpha_domain = st.text_input("Alpha Domain", "-2, 2")
omega_domain = st.text_input("Omega Domain", "-2, 2")
F_s_domain = st.text_input("|F(s)| Domain", "0, 2.5")
n = st.number_input("Grid Size", value=50, min_value=10)
lines = st.checkbox("Lines?", value=True)

alpha_domain = [float(item) for item in alpha_domain.split(",")]
omega_domain = [float(item) for item in omega_domain.split(",")]
F_s_domain = [float(item) for item in F_s_domain.split(",")]

if st.button("Plot"):
    try:
        fig =     compute_and_plot(func_str, alpha_domain, omega_domain,
                             F_s_domain, n, lines)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")


