import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import requests
import json

# Firebase Database URL
db_url = "https://digitaltwin-8ae1d-default-rtdb.firebaseio.com/"

def fetch_inventory_data(user_id, bin_id):
    """Fetch inventory data for a specific user and bin."""
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

def update_inventory_data(user_id, bin_id, inventory_data):
    """Update inventory data for a specific user and bin."""
    path = f"users/{user_id}/{bin_id}/inventory.json"
    response = requests.put(f"{db_url}{path}", data=json.dumps(inventory_data))
    if not response.ok:
        st.error(f"Failed to update data: {response.text}")

def calculate_bin_capacity(diameter, height):
    return np.pi * (diameter / 2) ** 2 * height

def create_bin_visualization(diameter, height, inventory):
    """Create a 3D visualization of the bin's moisture content."""
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

st.title("Grain Storage Bin Digital Twin")
user_id = st.sidebar.selectbox("Select User ID", ["User1", "User2"])
selected_bin = st.sidebar.selectbox("Select Bin ID", ["Bin1", "Bin2", "Bin3", "Bin4"])

inventory_df = fetch_inventory_data(user_id, selected_bin)
if inventory_df.empty:
    inventory_df = pd.DataFrame(columns=['Date', 'Commodity', 'Mass_tonnes', 'Test_Weight_kg_m3', 'Moisture_Content_percent', 'Height_m'])

bin_diameter = st.number_input("Bin Diameter (m):", value=10.0, min_value=1.0, step=0.5)
bin_height = st.number_input("Bin Height (m):", value=20.0, min_value=1.0, step=0.5)

if not inventory_df.empty:
    st.write("Current Inventory:", inventory_df)
    bin_fig = create_bin_visualization(bin_diameter, bin_height, inventory_df)
    st.plotly_chart(bin_fig)
else:
    st.error("No inventory data available for the selected bin.")

with st.form("inventory_form"):
    st.subheader("Add Grain to Inventory")
    commodity = st.selectbox("Commodity", ["Wheat", "Corn", "Oats", "Barley", "Canola", "Soybeans", "Rye"])
    mass = st.number_input("Mass (tonnes):", min_value=0.0, format='%.2f')
    test_weight = st.number_input("Test Weight (kg/m³):", min_value=0.0, format='%.2f')
    moisture_content = st.number_input("Moisture Content (%):", min_value=0.0, format='%.2f')
    submit_button = st.form_submit_button(label='Add Grain')

    if submit_button:
        if test_weight > 0:  # Check for valid test_weight to avoid ZeroDivisionError
            height_m = mass * 1000 / (np.pi * (bin_diameter / 2) ** 2 * test_weight)
            new_grain_data = {
                'Date': datetime.datetime.now().isoformat(),
                'Commodity': commodity,
                'Mass_tonnes': mass,
                'Test_Weight_kg_m3': test_weight,
                'Moisture_Content_percent': moisture_content,
                'Height_m': height_m
            }
            inventory_df = inventory_df.append(new_grain_data, ignore_index=True)
            update_inventory_data(user_id, selected_bin, inventory_df.to_dict(orient='records'))
            st.success("Inventory updated successfully.")
        else:
            st.error("Test weight must be greater than zero to calculate the height.")

# Display the updated inventory and possibly refresh the visualization
st.subheader("Updated Inventory")
st.write(inventory_df)

# If we have a non-empty DataFrame, update the visualization
if not inventory_df.empty:
    updated_bin_fig = create_bin_visualization(bin_diameter, bin_height, inventory_df)
    st.plotly_chart(updated_bin_fig)

# Future state section as a placeholder
st.subheader("Potential Future State")
st.write("This section will display the potential future state of the grain storage bin based on historical data and predictive models.")

