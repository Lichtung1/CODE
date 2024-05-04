import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go

def calculate_bin_capacity(diameter, height):
    return np.pi * (diameter / 2) ** 2 * height

def update_inventory(inventory, new_grain_data):
    inventory = pd.concat([inventory, new_grain_data], ignore_index=True)
    return inventory

def create_bin_visualization(diameter, height, inventory):
    # Create a cylindrical mesh for the bin
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, height, 100)
    theta, z = np.meshgrid(theta, z)
    x = (diameter / 2) * np.cos(theta)
    y = (diameter / 2) * np.sin(theta)

    # Calculate the cumulative height of the grain layers
    grain_heights = inventory['Height (m)'].cumsum()

    # Create a heatmap based on moisture data
    moisture_values = inventory['Moisture Content (%)'].values
    moisture_heatmap = np.zeros((100, 100))
    for i in range(len(moisture_values)):
        if i == 0:
            moisture_heatmap[:int(grain_heights[i] / height * 100), :] = moisture_values[i]
        else:
            moisture_heatmap[int(grain_heights[i-1] / height * 100):int(grain_heights[i] / height * 100), :] = moisture_values[i]

    # Set moisture content outside the grain layers to transparent
    moisture_heatmap[int(grain_heights[-1] / height * 100):, :] = np.nan

    # Create the 3D figure
    fig = go.Figure(data=[
        go.Surface(x=x, y=y, z=z, surfacecolor=moisture_heatmap, colorscale='Viridis', colorbar=dict(title='Moisture Content (%)'))
    ])

    # Add a transparent outer shell to show the structure of the bin
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(128,128,128,0.2)']]))

    fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Height (m)'),
                      title='Grain Storage Bin Moisture Content')

    return fig
    
def create_empty_bin_visualization(diameter, height):
    # Create a cylindrical mesh for the bin
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, height, 100)
    theta, z = np.meshgrid(theta, z)
    x = (diameter / 2) * np.cos(theta)
    y = (diameter / 2) * np.sin(theta)

    # Create the 3D figure
    fig = go.Figure(data=[
        go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(128,128,128,0.2)']])
    ])

    fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Height (m)'),
                      title='Empty Grain Storage Bin')

    return fig

# Streamlit UI
st.title("Grain Storage Bin Digital Twin")

# Bin selection dropdown
selected_bin = st.selectbox("Select Bin", ["Bin 1"])  # Start with only one bin

# Bin dimensions
bin_diameter = st.number_input("Bin Diameter (m):", value=10.0)
bin_height = st.number_input("Bin Height (m):", value=20.0)
bin_capacity_volume = calculate_bin_capacity(bin_diameter, bin_height)

# Initialize inventory dataframe for the selected bin
if f"inventory_{selected_bin}" not in st.session_state:
    st.session_state[f"inventory_{selected_bin}"] = pd.DataFrame(columns=['Date', 'Commodity', 'Mass (tonnes)', 'Test Weight (kg/m³)', 'Moisture Content (%)', 'Height (m)'])

inventory = st.session_state[f"inventory_{selected_bin}"]

# Grain input form
with st.form(key='grain_input_form'):
    st.subheader("Add Grain to Inventory")
    commodity = st.selectbox("Commodity", ["Wheat", "Corn", "Oats", "Barley", "Canola", "Soybeans", "Rye"])
    mass = st.number_input("Mass (tonnes):")
    test_weight = st.number_input("Test Weight (kg/m³):")
    moisture_content = st.number_input("Moisture Content (%):")
    submit_button = st.form_submit_button(label='Add Grain')

    if submit_button:
        volume = mass * 1000 / test_weight  # Convert mass to volume
        height = volume / (np.pi * (bin_diameter / 2) ** 2)  # Calculate the height of the added grain layer
        new_grain_data = pd.DataFrame({
            'Date': [datetime.date.today()],
            'Commodity': [commodity],
            'Mass (tonnes)': [mass],
            'Test Weight (kg/m³)': [test_weight],
            'Moisture Content (%)': [moisture_content],
            'Height (m)': [height]
        })
        inventory = update_inventory(inventory, new_grain_data)
        st.session_state[f"inventory_{selected_bin}"] = inventory

# Display bin capacity
st.subheader("Bin Capacity")
st.write(f"Bin Capacity (Volume): {bin_capacity_volume:.2f} m³")
bin_capacity_mass = bin_capacity_volume * test_weight / 1000  # Convert volume to mass
st.write(f"Bin Capacity (Mass): {bin_capacity_mass:.2f} tonnes")

# Display current inventory
st.subheader("Current Inventory")
st.write(inventory)

# 3D view of the bin with moisture content
st.subheader("Bin Moisture Content Visualization")
if not inventory.empty:
    bin_fig = create_bin_visualization(bin_diameter, bin_height, inventory)
    st.plotly_chart(bin_fig)
else:
    empty_bin_fig = create_empty_bin_visualization(bin_diameter, bin_height)
    st.plotly_chart(empty_bin_fig)

# Potential future state (not implemented in this mock version)
st.subheader("Potential Future State")
st.write("This section will display the potential future state of the grain storage bin based on historical data and predictive models.")
