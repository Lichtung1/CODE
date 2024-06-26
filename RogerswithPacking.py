import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import hyp2f1

def rogersTest(phi, mu, b, h, rho, g, numX, numY, binHeight):
    # Convert degrees to radians
    phi = np.radians(phi)

    x = np.linspace(0, b, numX)
    y = np.linspace(0, binHeight, numY)

    # Define A, B
    A, B = getAB(x, h, mu, phi, b)

    # Shear function calculation: Step 1. in Zhang & Britton (1998)
    S_1 = Sf_1(y, rho, b, A, B, g)
    S_1_prime = Sf_1_prime(y, rho, b, A, B, g)

    # Step 2. in Zhang & Britton (1998)
    tauXY = shear_xy(x, b, h, S_1)
    sigmaX_A = sigma_x(mu, x, b, S_1, S_1_prime, h)

    # Step 3. in Zhang & Britton (1998)
    beta = tauXY / sigmaX_A

    # Step 4. in Zhang & Britton (1998)
    sigmaY_A, alpha = sigma_y(mu, phi, b, x, S_1, S_1_prime, beta, h)

    # Janssen!
    pressYJans = janssenVert(phi, b*2, y, mu, g, rho)
    pressXJans = janssenHori(phi, mu, pressYJans)

    return x, y, sigmaX_A, sigmaY_A, pressXJans, pressYJans

def shear_xy(x, b, h, shear_func):
    result = np.zeros((len(shear_func), len(x)))
    for i in range(len(shear_func)):
        result[i, :] = (x/b)**h * shear_func[i]
    return result

def Sf_1(y, rho, b, A, B, g):
    delta_1 = B/A * np.sqrt(1 - A/B**2)
    return g * rho * b * (1 - (np.cosh(delta_1*y) + np.sinh(delta_1*y)/np.sqrt(1 - A/B**2)) * np.exp(-y*B/A))

def Sf_1_prime(y, rho, b, A, B, g):
    delta_1 = B/A * np.sqrt(1 - A/B**2)
    return g * rho * b * (-(delta_1*np.sinh(delta_1*y) + delta_1*np.cosh(delta_1*y)/np.sqrt(1 - A/B**2)) * np.exp(-y*B/A) + 
                          (np.cosh(delta_1*y) + np.sinh(delta_1*y)/np.sqrt(1 - A/B**2)) * B * np.exp(-y*B/A)/A)

def sigma_x(mu, x, b, S, S_prime, h):
    result = np.zeros((len(S), len(x)))
    for i in range(len(S)):
        result[i, :] = S[i]/mu + (b**(h+1) - x**(h+1)) * S_prime[i] / (b**h * (h+1))
    return result

def sigma_y(mu, phi, b, x, S, S_prime, beta, h):
    result = np.zeros((len(S), len(x)))
    alpha = 1/np.cos(phi)**2 * ((1 + np.sin(phi)**2) + 2*np.sin(phi)*np.sqrt(1 - beta**2/np.tan(phi)**2))
    for i in range(len(S)):
        result[i, :] = alpha[i, :] * (S[i]/mu + 1/(b**h * (h+1)) * S_prime[i] * (b**(h+1) - x**(h+1)))
    return result, alpha

def getAB(x, h, mu, phi, b):
    def integrand(x):
        return x**(h+1) * np.sqrt(1 - (x/b)**(h+1) * mu**2/np.tan(phi)**2)
    
    q, _ = quad(integrand, 0, b)
    
    z = mu**2 / np.tan(phi)**2
    F = hyp2f1(0.5, 1/(2*h), (2*h+1)/(2*h), z)
    
    B = b / (2*mu*np.cos(phi)**2) * (1 + np.sin(phi)**2 + 2*np.sin(phi)/((h+1)*np.sqrt(1 - mu**2/np.tan(phi)**2)) *
                                     (1 - mu**2/np.tan(phi)**2 + h*F*np.sqrt(1 - mu**2/np.tan(phi)**2)))
    
    A = 2*mu*b/(h+1)*B - 1/(b**h * (h+1) * np.cos(phi)**2) * (b**(h+2)/(h+2) * (1 + np.sin(phi)**2) + 2*np.sin(phi)*q)
    
    return A, B

def janssenVert(phi, binDia, y, mu, g, rho):
    k = (1 - np.sin(phi)) / (1 + np.sin(phi))
    hydR = binDia / 2
    return rho * g * hydR / (mu * k) * (1 - np.exp(-y * mu * k / hydR))

def janssenHori(phi, mu, pressYJans):
    k = (1 - np.sin(phi)) / (1 + np.sin(phi))
    return pressYJans * k

def changeInDensityByPressure(P, params, MC):
    return params[0] * P + params[1] * np.sqrt(P) + params[2] * P * MC

# Streamlit app
st.title('Rogers Test Visualization with Packing Calculations')

# Input parameters using text input boxes
col1, col2 = st.columns(2)

with col1:
    phi = st.number_input('Phi (degrees)', min_value=0.0, max_value=90.0, value=22.0, step=0.1)
    mu = st.number_input('Mu', min_value=0.0, max_value=1.0, value=0.36, step=0.01)
    b = st.number_input('Bin radius (m)', min_value=0.1, max_value=10.0, value=2.5, step=0.1)

with col2:
    h = st.number_input('h', min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    rho0 = st.number_input('Initial Density (kg/m^3)', min_value=500, max_value=2000, value=834, step=1)
    binHeight = st.number_input('Bin Height (m)', min_value=1, max_value=20, value=10, step=1)

# Fixed parameters
g = 9.81  # m/s^2
numX = 101  # Number of X points
numY = 101  # Number of Y points

# Dropdown for commodity selection
commodity = st.selectbox('Select commodity:', ['hard wheat', 'soft wheat', 'corn', 'rice'])

# Moisture content input
MC = st.number_input('Moisture Content (%)', min_value=0.0, max_value=100.0, value=12.0, step=0.1)

# Run button
if st.button('Run Calculation'):
    # Run the Rogers test calculation
    x, y, sigmaX_A, sigmaY_A, pressXJans, pressYJans = rogersTest(phi, mu, b, h, rho0, g, numX, numY, binHeight)

    # Select parameters based on commodity
    if commodity == 'hard wheat':
        params = np.array([-0.3143995, 1.126975113, 0.01655364])
    elif commodity == 'soft wheat':
        params = np.array([-0.8034, 8.0876, 0.039415])
    elif commodity == 'corn':
        params = np.array([-1.29692739970224, 7.11178006395763, 0.0787013837832110])
    elif commodity == 'rice':
        params = np.array([-0.7580, 8.9355, 0.0499])

    # Calculate density
    density = np.zeros_like(sigmaY_A)
    center_data = []  # List to store center line data

    for j in range(numY):
        for i in range(numX):
            pressure_kPa = sigmaY_A[j, i] / 1000  # Convert Pa to kPa
            delta = changeInDensityByPressure(pressure_kPa, params, MC)
            density[j, i] = rho0 + delta
            
            if i == 0:  # This is the center line
                center_data.append((y[j], pressure_kPa, delta))

    # Plot center line data
    fig, ax = plt.subplots(figsize=(10, 6))
    depths, pressures, deltas = zip(*center_data)
    ax.plot(pressures, deltas, 'b-')
    ax.set_xlabel("Pressure (kPa)")
    ax.set_ylabel("Change in Density (kg/m³)")
    ax.set_title("Change in Density vs Pressure at the Center of the Bin")
    ax.grid(True)
    st.pyplot(fig)

    # Create heatmap plots
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))

    im1 = axs[0].imshow(sigmaX_A/1000, extent=[0, b, binHeight, 0], aspect='auto', cmap='viridis', origin='upper')
    axs[0].set_title('Horizontal Stress σx')
    axs[0].set_xlabel('Distance from center (m)')
    axs[0].set_ylabel('Depth from top (m)')
    plt.colorbar(im1, ax=axs[0], label='Stress (kPa)')
    
    im2 = axs[1].imshow(sigmaY_A/1000, extent=[0, b, binHeight, 0], aspect='auto', cmap='viridis', origin='upper')
    axs[1].set_title('Vertical Stress σy')
    axs[1].set_xlabel('Distance from center (m)')
    axs[1].set_ylabel('Depth from top (m)')
    plt.colorbar(im2, ax=axs[1], label='Stress (kPa)')
    
    im3 = axs[2].imshow(density, extent=[0, b, binHeight, 0], aspect='auto', cmap='viridis', origin='upper')
    axs[2].set_title('Density')
    axs[2].set_xlabel('Distance from center (m)')
    axs[2].set_ylabel('Depth from top (m)')
    plt.colorbar(im3, ax=axs[2], label='Density (kg/m^3)')

    plt.tight_layout()
    st.pyplot(fig)

    # Display average, min, and max density
    avgDensity = np.mean(density)
    st.write(f'Average Bulk Density: {avgDensity:.2f} kg/m^3')
    st.write(f"Minimum density: {np.min(density):.2f} kg/m^3")
    st.write(f"Maximum density: {np.max(density):.2f} kg/m^3")
