import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go

def calculate_bin_capacity(diameter, height):
    return np.pi * (diameter / 2) ** 2 * height

def update_inventory(inventory, new_grain_data):
    if inventory.empty:
        inventory = new_grain_data
    else:
        inventory = pd.concat([inventory, new_grain_data], ignore_index=True)
    return inventory

def create_bin_visualization(diameter, height, moisture_data):
    # Create a cylindrical mesh for the bin
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, height, 100)
    theta, z = np.meshgrid(theta, z)
    x = (diameter / 2) * np.cos(theta)
    y = (diameter / 2) * np.sin(theta)

    # Create a heatmap based on moisture data
    moisture_values = moisture_data['Moisture Content (%)'].values
    moisture_heatmap = np.zeros((100, 100))
    moisture_heatmap[:len(moisture_values), :] = moisture_values.reshape(-1, 1)

    # Create the 3D figure
    fig = go.Figure(data=[
        go.Surface(x=x, y=y, z=z, surfacecolor=moisture_heatmap, colorscale='Viridis', colorbar=dict(title='Moisture Content (%)'))
    ])

    fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Height (m)'),
                      title='Grain Storage Bin Moisture Content')

    return fig

# Streamlit UI
st.title("Grain Storage Bin Digital Twin")

# Bin dimensions
bin_diameter = st.number_input("Bin Diameter (m):", value=10.0)
bin_height = st.number_input("Bin Height (m):", value=20.0)
bin_capacity = calculate_bin_capacity(bin_diameter, bin_height)

# Initialize inventory dataframe
inventory_data = {
    'Date': [],
    'Grain Type': [],
    'Volume (m³)': [],
    'Test Weight (kg/m³)': [],
    'Moisture Content (%)': []
}
inventory = pd.DataFrame(inventory_data)

# Grain input form
with st.form(key='grain_input_form'):
    st.subheader("Add Grain to Inventory")
    grain_type = st.text_input("Grain Type:")
    volume = st.number_input("Volume (m³):")
    test_weight = st.number_input("Test Weight (kg/m³):")
    moisture_content = st.number_input("Moisture Content (%):")
    submit_button = st.form_submit_button(label='Add Grain')

    if submit_button:
        new_grain_data = {
            'Date': [datetime.date.today()],
            'Grain Type': [grain_type],
            'Volume (m³)': [volume],
            'Test Weight (kg/m³)': [test_weight],
            'Moisture Content (%)': [moisture_content]
        }
        inventory = update_inventory(inventory, pd.DataFrame(new_grain_data))

# Display bin capacity
st.subheader("Bin Capacity")
st.write(f"Bin Capacity: {bin_capacity:.2f} m³")

# Display current inventory
st.subheader("Current Inventory")
st.write(inventory)

# 3D view of the bin with moisture content
st.subheader("Bin Moisture Content Visualization")
if not inventory.empty:
    bin_fig = create_bin_visualization(bin_diameter, bin_height, inventory)
    st.plotly_chart(bin_fig)
else:
    st.write("No grain data available for visualization.")

# Potential future state (not implemented in this mock version)
st.subheader("Potential Future State")
st.write("This section will display the potential future state of the grain storage bin based on historical data and predictive models.")
