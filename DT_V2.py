import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import requests
import json
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import hyp2f1

# Firebase Database URL
db_url = "https://digitaltwin-8ae1d-default-rtdb.firebaseio.com/"

# Digital Twin Function
def fetch_inventory_data(user_id, bin_id): #get inforation from database
    path = f"users/{user_id}/{bin_id}/inventory.json"
    response = requests.get(f"{db_url}{path}")
    if response.ok:
        data = response.json()
        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    else:
        st.error(f"Failed to fetch data: {response.text}")
        return pd.DataFrame()

def update_inventory_data(user_id, bin_id, inventory_data): #update inventory data for a specific user and bin
    path = f"users/{user_id}/{bin_id}/inventory.json"
    response = requests.put(f"{db_url}{path}", data=json.dumps(inventory_data))
    if not response.ok:
        st.error(f"Failed to update data: {response.text}")

def calculate_bin_capacity(diameter, height):
    return np.pi * (diameter / 2) ** 2 * height

def create_bin_visualization(diameter, height, inventory): #creat a 3D visualization of the the specific bin's moisture content
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, height, 100)
    theta, z = np.meshgrid(theta, z)
    x = (diameter / 2) * np.cos(theta)
    y = (diameter / 2) * np.sin(theta)
    moisture_heatmap = np.zeros((100, 100))
    if not inventory.empty:
        grain_heights = inventory['Height_m'].cumsum()
        moisture_values = inventory['Moisture_Content_percent']
        for i in range(len(moisture_values)):
            start_index = int((grain_heights[i-1] / height) * 100) if i else 0
            end_index = int((grain_heights[i] / height) * 100)
            moisture_heatmap[start_index:end_index, :] = moisture_values[i]
        custom_colorscale = [[0.0, 'rgba(128,128,128,1)'], [9/30, 'green'], [14/30, 'yellow'], [20/30, 'red'], [1.0, 'red']]
        fig = go.Figure(data=go.Surface(x=x, y=y, z=z, surfacecolor=moisture_heatmap, colorscale=custom_colorscale, cmin=0, cmax=30, colorbar=dict(title='Moisture Content (%)')))
        fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(128,128,128,0.2)']]))
        fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Height (m)'), title='Grain Storage Bin Moisture Content')
        return fig
    return None

def clean_inventory_data(user_id, bin_id):
    inventory_df = fetch_inventory_data(user_id, bin_id)
    if not inventory_df.empty:
        # Remove rows where both Height_m and Mass_tonnes are 0
        inventory_df = inventory_df[(inventory_df['Height_m'] != 0) | (inventory_df['Mass_tonnes'] != 0)]
        update_inventory_data(user_id, bin_id, inventory_df.to_dict(orient='records'))

def unload_grain(user_id, bin_id, mass_to_unload, bin_diameter):
    inventory_df = fetch_inventory_data(user_id, bin_id)
    total_mass = inventory_df['Mass_tonnes'].sum()
    if mass_to_unload > total_mass:
        st.warning(f"Not enough grain to unload. Short by {mass_to_unload - total_mass:.2f} tonnes.")
        return

    remaining_mass = mass_to_unload
    preserved_rows = []

    for index, row in inventory_df.iterrows():
        if remaining_mass <= 0:
            preserved_rows.append(row)
            continue

        if remaining_mass >= row['Mass_tonnes']:
            remaining_mass -= row['Mass_tonnes']
            continue
        else:
            row['Mass_tonnes'] -= remaining_mass
            row['Height_m'] = row['Mass_tonnes'] * 1000 / (row['Test_Weight_kg_m3'] * np.pi * (bin_diameter / 2) ** 2)
            preserved_rows.append(row)
            remaining_mass = 0

    new_inventory = pd.DataFrame(preserved_rows)
    update_inventory_data(user_id, bin_id, new_inventory.to_dict(orient='records'))
    clean_inventory_data(user_id, bin_id)  # Clean up after unloading

# Packing Function
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
st.title("Grain Storage Bin Digital Twin with Pressure Model")

# Sidebar for user and bin selection, and bin dimensions
st.sidebar.header("Bin Selection and Dimensions")
user_id = st.sidebar.selectbox("Select User ID", ["User1", "User2"])
selected_bin = st.sidebar.selectbox("Select Bin ID", ["Bin1", "Bin2", "Bin3", "Bin4"])
bin_diameter = st.sidebar.number_input("Bin Diameter (m):", value=10.0, min_value=0.1, step=0.1)
bin_height = st.sidebar.number_input("Bin Height (m):", value=20.0, min_value=0.1, step=0.1)

# Fetch the current inventory
inventory_df = fetch_inventory_data(user_id, selected_bin)
if inventory_df.empty:
    inventory_df = pd.DataFrame(columns=['Date', 'Commodity', 'Mass_tonnes', 'Test_Weight_kg_m3', 'Moisture_Content_percent', 'Height_m'])

# Define commodities with available compaction data
available_commodities = ["Hard Wheat", "Soft Wheat", "Corn", "Rice"]
total_grain_height = inventory_df['Height_m'].sum()

# Grain input form
with st.form("inventory_form"):
    st.subheader("Add Grain to Inventory")
    commodity = st.selectbox("Commodity", available_commodities)
    mass = st.number_input("Mass (tonnes):", min_value=0.0, format='%.2f')
    test_weight = st.number_input("Test Weight (kg/m³):", min_value=0.0, format='%.2f')
    moisture_content = st.number_input("Moisture Content (%):", min_value=0.0, format='%.2f')
    submit_button = st.form_submit_button(label='Add Grain')

    # In the main Streamlit app:
    if submit_button:
        if test_weight > 0 and mass > 0:  # Add check for positive mass
            height_m = mass * 1000 / (np.pi * (bin_diameter / 2) ** 2 * test_weight)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            new_grain_data = {
                'Date': current_time,
                'Commodity': commodity,
                'Mass_tonnes': mass,
                'Test_Weight_kg_m3': test_weight,
                'Moisture_Content_percent': moisture_content,
                'Height_m': height_m
            }
            new_grain_df = pd.DataFrame([new_grain_data])
            inventory_df = pd.concat([inventory_df, new_grain_df], ignore_index=True)
            update_inventory_data(user_id, selected_bin, inventory_df.to_dict(orient='records'))
            clean_inventory_data(user_id, selected_bin)  # Clean up after adding
            st.success("Inventory updated successfully.")
        else:
            st.error("Test weight and mass must be greater than zero to calculate the height.")
    
    
# Unload grain section
st.sidebar.subheader("Unload Grain")
unload_mass = st.sidebar.number_input("Mass to unload (tonnes):", min_value=0.0, format='%.2f')
unload_button = st.sidebar.button("Unload Grain")

if unload_button:
    unload_grain(user_id, selected_bin, unload_mass, bin_diameter)
    inventory_df = fetch_inventory_data(user_id, selected_bin)  # Fetch updated inventory data after unloading
    clean_inventory_data(user_id, selected_bin)  # Clean up on load
    
# Display the current and possibly updated inventory
st.subheader("Current Inventory")
st.write(inventory_df)

# Display the bin visualization
st.subheader("Bin Visualization")
if not inventory_df.empty:
    bin_fig = create_bin_visualization(bin_diameter, bin_height, inventory_df)
    st.plotly_chart(bin_fig)

# Pressure Model Section
st.subheader("Pressure Model and Density Change")

# Input parameters for pressure model
col1, col2 = st.columns(2)

with col1:
    phi = st.number_input('Angle of internal friction (degrees)', min_value=0.0, max_value=90.0, value=22.0, step=0.1)
    mu = st.number_input('Coefficient of friction', min_value=0.0, max_value=1.0, value=0.36, step=0.01)

with col2:
    h = st.number_input('Rogers factor', min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    rho0 = st.number_input('Initial Density (kg/m^3)', min_value=500, max_value=2000, value=834, step=1)

# Fixed parameters
g = 9.81  # m/s^2
numX = 101  # Number of X points
numY = 101  # Number of Y points

  
# Before the pressure model calculation
center_data = []  # List to store center line data for each layer

# Inside the pressure model calculation loop
if st.button('Run Pressure Model'):
    if inventory_df.empty:
        st.warning("No inventory data available. Please add grain to the inventory first.")
    else:
        # Run the Rogers test calculation
        x, y, sigmaX_A, sigmaY_A, pressXJans, pressYJans = rogersTest(phi, mu, bin_diameter/2, h, rho0, g, numX, numY, total_grain_height)

        # Calculate density for each layer
        density_layers = []
        current_height = 0
        
        for index, row in inventory_df.iterrows():
            layer_height = row['Height_m']
            layer_start = int(current_height / bin_height * numY)
            layer_end = int((current_height + layer_height) / bin_height * numY)
            
            # Use Test Weight as initial density and Moisture Content from the database
            initial_density = row['Test_Weight_kg_m3']
            moisture_content = row['Moisture_Content_percent']
            
            # Select appropriate parameters based on commodity
            if row['Commodity'].lower() == 'hard wheat':
                params = np.array([-0.488, 6.59, 0.0203])
            elif row['Commodity'].lower() == 'soft wheat':
                params = np.array([-0.8034, 8.0876, 0.039415])
            elif row['Commodity'].lower() == 'corn':
                params = np.array([-1.29692739970224, 7.11178006395763, 0.0787013837832110])
            elif row['Commodity'].lower() == 'rice':
                params = np.array([-0.7580, 8.9355, 0.0499])
            else:
                st.warning(f"No compaction data available for {row['Commodity']}. Using default parameters.")
                params = np.array([-0.488, 6.59, 0.0203])  # default to hard wheat parameters
        
            layer_density = np.zeros((layer_end - layer_start, numX))
            for j in range(layer_end - layer_start):
                for i in range(numX):
                    pressure_kPa = sigmaY_A[layer_start + j, i] / 1000  # Convert Pa to kPa
                    delta = changeInDensityByPressure(pressure_kPa, params, moisture_content)
                    layer_density[j, i] = initial_density + delta
            
            density_layers.append(layer_density)
            current_height += layer_height
        
        # Concatenate all layers into one large 2D array
        density = np.vstack(density_layers)
        
        # Create a mask for the filled part of the bin
        filled_mask = density > 0

        # Extract the center column from the concatenated density array
        center_density = density[:, 0]  # Assuming 0 is the center column
        
        # Calculate the pressures at the center
        center_pressures = sigmaY_A[:len(center_density), 0] / 1000  # Convert Pa to kPa
        
        # Plot center line data
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the density
        ax.plot(center_pressures, center_density, '-', label='Density')
        
        # Add horizontal lines to show layer separations
        current_height = 0
        for index, row in inventory_df.iterrows():
            current_height += row['Height_m']
            layer_depth = bin_height - current_height
            ax.axhline(y=center_density[int(layer_depth / bin_height * len(center_density))],
                       color='r', linestyle='--')
            ax.text(max(center_pressures), center_density[int(layer_depth / bin_height * len(center_density))],
                    f" {row['Commodity']} - Layer {index+1}", verticalalignment='bottom', horizontalalignment='right')
        
        ax.set_xlabel("Pressure (kPa)")
        ax.set_ylabel("Density (kg/m³)")
        ax.set_title("Density vs Pressure at the Center of the Bin")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Create heatmap plots
        fig, axs = plt.subplots(1, 3, figsize=(20, 7))
        
        # X Pressure
        im1 = axs[0].imshow(sigmaX_A/1000, extent=[0, bin_diameter/2, total_grain_height, 0], aspect='auto', cmap='viridis', origin='upper')
        axs[0].set_title('Horizontal Stress σx')
        axs[0].set_xlabel('Distance from center (m)')
        axs[0].set_ylabel('Depth from top (m)')
        plt.colorbar(im1, ax=axs[0], label='Stress (kPa)')
        
        # Y Pressure
        im2 = axs[1].imshow(sigmaY_A/1000, extent=[0, bin_diameter/2, total_grain_height, 0], aspect='auto', cmap='viridis', origin='upper')
        axs[1].set_title('Vertical Stress σy')
        axs[1].set_xlabel('Distance from center (m)')
        axs[1].set_ylabel('Depth from top (m)')
        plt.colorbar(im2, ax=axs[1], label='Stress (kPa)')
        
        # Density plot
        masked_density = np.ma.masked_where(~filled_mask, density)
        im3 = axs[2].imshow(masked_density, extent=[0, bin_diameter/2, total_grain_height, 0], aspect='auto', cmap='viridis', origin='upper')
        axs[2].set_title('Density')
        axs[2].set_xlabel('Distance from center (m)')
        axs[2].set_ylabel('Depth from top (m)')
        cbar = plt.colorbar(im3, ax=axs[2], label='Density (kg/m³)')
        cbar.set_ticks(np.linspace(masked_density.min(), masked_density.max(), 10))
        
        # Add horizontal lines to separate layers
        current_height = 0
        for index, row in inventory_df.iterrows():
            current_height += row['Height_m']
            axs[2].axhline(y=current_height, color='r', linestyle='--')
            axs[2].text(bin_diameter/4, current_height - row['Height_m']/2, f"{row['Commodity']} - Layer {index+1}", 
                        verticalalignment='center', horizontalalignment='center')
        
        axs[2].set_ylim(total_grain_height, 0)  # Set y-axis limits to match bin height
        axs[2].set_ylabel('Depth from top (m)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display average, min, and max density (only for the filled part)
        filled_density = density[filled_mask]
        avgDensity = np.mean(filled_density)
        st.write(f'Average Bulk Density: {avgDensity:.2f} kg/m^3')
        st.write(f"Minimum density: {np.min(filled_density):.2f} kg/m^3")
        st.write(f"Maximum density: {np.max(filled_density):.2f} kg/m^3")
