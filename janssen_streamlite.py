import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def janssen_equation(rho, g, R, k, mu, h):
    return rho * g * R / (k * mu) * (1 - np.exp(-k * mu * h / R))

def plot_janssen_equation(rho, g, R, k, mu, h_max):
    h_vals = np.linspace(0, h_max, 100)
    pressure_vals = janssen_equation(rho, g, R, k, mu, h_vals)

    fig, ax = plt.subplots()
    ax.plot(pressure_vals, h_vals)
    ax.set_xlabel('Pressure (Pa)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Janssen Equation')
    ax.grid(True)

    return fig

# Streamlit UI
st.title("Janssen Equation Plotter")

rho = st.number_input("Density (kg/m^3):", value=1000.0)
g = st.number_input("Acceleration due to gravity (m/s^2):", value=9.81)
R = st.number_input("Bin radius (m):", value=1.0)
k = st.number_input("Janssen coefficient:", value=0.4)
mu = st.number_input("Coefficient of friction:", value=0.3)
h_max = st.number_input("Maximum height (m):", value=10.0)

if st.button('Plot'):
    try:
        fig = plot_janssen_equation(rho, g, R, k, mu, h_max)
        st.pyplot(fig)
    except Exception as e:
        st.error(str(e))
