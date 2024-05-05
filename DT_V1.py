import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import requests
import json

# Firebase Database URL
db_url = "https://digitaltwin-8ae1d-default-rtdb.firebaseio.com/"

# Function to fetch inventory data using HTTP requests
def fetch_inventory_data(user_id, bin_id):
    """Fetch inventory data for a specific user and bin."""
    path = f"users/{user_id}/{bin_id}/inventory.json"
    response = requests.get(f"{db_url}{path}")
    if response.ok:
        data = response.json()
        if data:
            # Since data is a list of dictionaries, we can directly convert it to a DataFrame
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no data
    else:
        st.error(f"Failed to fetch data: {response.text}")
        return pd.DataFrame()
        
# Function to update inventory data using HTTP requests
def update_inventory_data(user_id, bin_id, inventory_data):
    """Update inventory data for a specific user and bin."""
    path = f"users/{user_id}/{bin_id}/inventory.json"
    response = requests.put(f"{db_url}{path}", data=json.dumps(inventory_data))
    if not response.ok:
        st.error(f"Failed to update data: {response.text}")

# Function to calculate bin capacity
def calculate_bin_capacity(diameter, height):
    return np.pi * (diameter / 2) ** 2 * height

# Function to create visualization of the bin
def create_bin_visualization(diameter, height, inventory):
    # Create a cylindrical mesh for the bin
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, height, 100)
    theta, z = np.meshgrid(theta, z)
    x = (diameter / 2) * np.cos(theta)
    y = (diameter / 2) * np.sin(theta)

    # Create a heatmap based on moisture data
    moisture_heatmap = np.zeros((100, 100))

    if not inventory.empty:
        grain_heights = inventory['Height_m'].cumsum()
        moisture_values = inventory['Moisture_Content_percent']

        for i in range(len(moisture_values)):
            start_index = 0 if i == 0 else min(int(grain_heights[i-1] / height * 100), 99)
            end_index = min(int(grain_heights[i] / height * 100), 100)
            moisture_heatmap[start_index:end_index, :] = moisture_values[i]

        # Set moisture content outside the grain layers to transparent
        if len(grain_heights) > 0:
            last_grain_height_index = min(int(grain_heights.iloc[-1] / height * 100), 99)
            moisture_heatmap[last_grain_height_index+1:, :] = np.nan

    # Define the position of the colors in terms of the normalized scale (0 to 1)
    custom_colorscale = [
        [0.0, 'rgba(128,128,128,1)'],  # Grey color for 0
        [9/30, 'green'],                # Green color at MC of 9
        [14/30, 'yellow'],              # Yellow color at MC of 14
        [20/30, 'red'],                 # Red color at MC of 20
        [1.0, 'red']                    # Use red color for the maximum value as well to avoid other colors
    ]
    fig = go.Figure(data=[
        go.Surface(
            x=x, 
            y=y, 
            z=z, 
            surfacecolor=moisture_heatmap, 
            colorscale=custom_colorscale,
            cmin=0,  # The minimum value for your moisture content
            cmax=30,  # The maximum value for your moisture content
            colorbar=dict(title='Moisture Content (%)')
        )
    ])
    # Add a transparent outer shell to show the structure of the bin
    fig.add_trace(go.Surface(
        x=x, 
        y=y, 
        z=z, 
        opacity=0.1, 
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(128,128,128,0.2)']]
    ))

    fig.update_layout(
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Height (m)'),
        title='Grain Storage Bin Moisture Content'
    )

    return fig

#Streamlit UI setup
st.title("Grain Storage Bin Digital Twin")

# User and bin selection (for simplicity, hardcoded here)
user_id = 'User1'
selected_bin = 'Bin1'

# Fetch inventory from Firebase
inventory_df = fetch_inventory_data(user_id, selected_bin)

# Check if the DataFrame is not empty before processing
if not inventory_df.empty:
    # Display the inventory data
    st.write("Current Inventory:", inventory_df)

    # Bin dimensions input (assuming some example static sizes for demo)
    bin_diameter = st.number_input("Bin Diameter (m):", value=10.0, min_value=1.0, step=0.5)
    bin_height = st.number_input("Bin Height (m):", value=20.0, min_value=1.0, step=0.5)

    # Generate and display the 3D visualization of the bin's moisture content
    bin_fig = create_bin_visualization(bin_diameter, bin_height, inventory_df)
    if bin_fig:
        st.plotly_chart(bin_fig)
else:
    st.error("No inventory data available for the selected bin.")

# Convert fetched data into a DataFrame
if isinstance(inventory, list):  # Check if data is in list format, which we expect
    inventory_df = pd.DataFrame(inventory)
else:
    inventory_df = pd.DataFrame()

# Bin dimensions input
bin_diameter = st.number_input("Bin Diameter (m):", value=10.0)
bin_height = st.number_input("Bin Height (m):", value=20.0)
bin_capacity_volume = calculate_bin_capacity(bin_diameter, bin_height)

# Display bin capacity
st.subheader("Bin Capacity")
st.write(f"Bin Capacity (Volume): {bin_capacity_volume:.2f} m³")
if not inventory_df.empty:
    test_weight = inventory_df['Test_Weight_kg_m3'].iloc[-1]  # Get the test weight of the last added grain
    bin_capacity_mass = bin_capacity_volume * test_weight / 1000  # Convert volume to mass assuming density is based on test weight
    st.write(f"Bin Capacity (Mass): {bin_capacity_mass:.2f} tonnes")
else:
    st.write("Bin Capacity (Mass): N/A")

# Grain input form
with st.form(key='grain_input_form'):
    st.subheader("Add Grain to Inventory")
    commodity = st.selectbox("Commodity", ["Wheat", "Corn", "Oats", "Barley", "Canola", "Soybeans", "Rye"])
    mass = st.number_input("Mass (tonnes):")
    test_weight = st.number_input("Test Weight (kg/m³):")
    moisture_content = st.number_input("Moisture Content (%):")
    submit_button = st.form_submit_button(label='Add Grain')

    if submit_button:
        new_grain_data = {
            'Date': str(datetime.date.today()),
            'Commodity': commodity,
            'Mass_tonnes': mass,
            'Test_Weight_kg_m3': test_weight,
            'Moisture_Content_percent': moisture_content,
            'Height_m': mass * 1000 / (np.pi * (bin_diameter / 2) ** 2 * test_weight)  # Calculate the height of the grain layer added
        }
        inventory_df = inventory_df.append(new_grain_data, ignore_index=True)
        inventory_data_for_firebase = inventory_df.to_dict(orient='records')
        update_inventory_data(user_id, selected_bin, inventory_data_for_firebase)

# Display current inventory
st.subheader("Current Inventory")
st.write(inventory_df)

# 3D view of the bin with moisture content
st.subheader("Bin Moisture Content Visualization")
if not inventory_df.empty:
    bin_fig = create_bin_visualization(bin_diameter, bin_height, inventory_df)
    st.plotly_chart(bin_fig)
else:
    empty_bin_fig = create_empty_bin_visualization(bin_diameter, bin_height)
    st.plotly_chart(empty_bin_fig)

# Future state section as a placeholder
st.subheader("Potential Future State")
st.write("This section will display the potential future state of the grain storage bin based on historical data and predictive models.")
