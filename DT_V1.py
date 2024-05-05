import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import requests
import json
import ast

# Firebase project ID and Database URL
project_id = "digitaltwin-8ae1d-default-rtdb"
db_url = f"https://{project_id}.firebaseio.com"

# Function Definitions
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
    moisture_heatmap = np.zeros((100, 100))
    
    if not inventory.empty and 'Height_m' in inventory.columns and 'Moisture_Content_percent' in inventory.columns:
        grain_heights = inventory['Height_m'].cumsum()
        moisture_values = inventory['Moisture_Content_percent'].values
        
        for i in range(len(moisture_values)):
            start_index = min(int(grain_heights[i-1] / height * 100), 100) if i != 0 else 0
            end_index = min(int(grain_heights[i] / height * 100), 100)
            moisture_heatmap[start_index:end_index, :] = moisture_values[i]
        
        if len(grain_heights) > 0:
            last_grain_height_index = min(int(grain_heights.iloc[-1] / height * 100), 99)
            moisture_heatmap[last_grain_height_index+1:, :] = np.nan
    
    fig = go.Figure(data=[
        go.Surface(x=x, y=y, z=z, surfacecolor=moisture_heatmap, colorscale='Viridis', colorbar=dict(title='Moisture Content (%)'))
    ])
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(128,128,128,0.2)']]))
    fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Height (m)'),
                      title='Grain Storage Bin Moisture Content')
    return fig

def create_empty_bin_visualization(diameter, height):
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, height, 100)
    theta, z = np.meshgrid(theta, z)
    x = (diameter / 2) * np.cos(theta)
    y = (diameter / 2) * np.sin(theta)
    
    fig = go.Figure(data=[
        go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(128,128,128,0.2)']])
    ])
    fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Height (m)'),
                      title='Empty Grain Storage Bin')
    return fig

def unload_grain(inventory, mass_to_unload):
    total_mass = inventory['Mass_tonnes'].sum()
    if mass_to_unload > total_mass:
        st.warning(f"Not enough grain to unload. Short by {mass_to_unload - total_mass:.2f} tonnes.")
        return inventory

    remaining_mass = mass_to_unload
    new_inventory = pd.DataFrame(columns=inventory.columns)

    for index, row in inventory.iloc[::-1].iterrows():
        if remaining_mass >= row['Mass_tonnes']:
            remaining_mass -= row['Mass_tonnes']
        else:
            new_row = row.copy()
            new_row['Mass_tonnes'] -= remaining_mass
            new_row['Height_m'] = new_row['Mass_tonnes'] * 1000 / (new_row['Test_Weight_kg_m3'] * np.pi * (bin_diameter / 2) ** 2)
            new_inventory = pd.concat([new_inventory, pd.DataFrame(new_row).T], ignore_index=True)
            break

    new_inventory = pd.concat([new_inventory, inventory.iloc[:index]], ignore_index=True)
    return new_inventory
    
def fix_dict_format(data):
    if isinstance(data, dict):
        return data  # If data is already a dictionary, return it as is
    elif isinstance(data, str):
        corrected = data.strip()
        if not corrected.endswith('}'):
            corrected += '}'
        corrected = corrected.replace('\n', '').replace('}', '').replace('{', '')
        corrected = "{" + ", ".join([item.strip() for item in corrected.split('" ')]) + "}"
        return corrected
    else:
        raise ValueError("Unsupported data type for fix_dict_format")

# Streamlit UI Setup
st.title("Grain Storage Bin Digital Twin")

# User authentication
user_id = st.text_input("Enter User ID")

if user_id:
    # Retrieve bins and inventory data for the user from Firebase using REST API
    bins_ref = f"{db_url}/users/{user_id}/bins.json"
    response = requests.get(bins_ref)
    if response.status_code == 200:
        bins_data = response.json() or {}
        st.session_state.bins = list(bins_data.keys())
    else:
        st.error("Failed to retrieve bins from Firebase.")
        st.stop()

    if not st.session_state.get('bins', []):
        st.warning("No bins found for the user.")
        st.stop()

    selected_bin = st.selectbox("Select Bin", st.session_state.bins)

    # Define bin dimensions
    bin_diameter = st.number_input("Bin Diameter (m):", value=10.0)
    bin_height = st.number_input("Bin Height (m):", value=20.0)

    # Initialize or retrieve existing inventory dataframe for the selected bin
    if f"inventory_{selected_bin}" not in st.session_state:
        inventory_ref = f"{db_url}/users/{user_id}/{selected_bin}/inventory.json"
        inventory_response = requests.get(inventory_ref)
        if inventory_response.status_code == 200:
            inventory_data = inventory_response.json() or []
            st.session_state[f"inventory_{selected_bin}"] = pd.DataFrame(inventory_data)
        else:
            st.session_state[f"inventory_{selected_bin}"] = pd.DataFrame()
            st.error(f"Failed to retrieve inventory for {selected_bin}.")

    inventory = st.session_state[f"inventory_{selected_bin}"]

    # Display current inventory
    st.subheader("Current Inventory")
    if not inventory.empty:
        st.dataframe(inventory)
    else:
        st.write("No inventory data available for the selected bin.")

    # Grain input form
    with st.form(key='grain_input_form'):
        st.subheader("Add Grain to Inventory")
        commodity = st.selectbox("Commodity", ["Wheat", "Corn", "Oats", "Barley", "Canola", "Soybeans", "Rye"])
        mass = st.number_input("Mass (tonnes):")
        test_weight = st.number_input("Test Weight (kg/m3):")
        moisture_content = st.number_input("Moisture Content (%):")
        submit_button = st.form_submit_button(label='Add Grain')

        if submit_button:
            volume = mass * 1000 / test_weight  # Convert mass to volume
            height = volume / (np.pi * (bin_diameter / 2) ** 2)  # Calculate the height of the added grain layer
            new_grain_data = pd.DataFrame({
                'Date': [datetime.date.today()],
                'Commodity': [commodity],
                'Mass_tonnes': [mass],
                'Test_Weight_kg_m3': [test_weight],
                'Moisture_Content_percent': [moisture_content],
                'Height_m': [height]
            })
            inventory = update_inventory(inventory, new_grain_data)
            st.session_state[f"inventory_{selected_bin}"] = inventory

    # Grain unload form
    with st.form(key='grain_unload_form'):
        st.subheader("Unload Grain from Inventory")
        mass_to_unload = st.number_input("Mass to Unload (tonnes):")
        unload_button = st.form_submit_button(label='Unload Grain')

        if unload_button:
            inventory = unload_grain(inventory, mass_to_unload)
            st.session_state[f"inventory_{selected_bin}"] = inventory
    # 3D view of the bin with moisture content
    st.subheader("Bin Moisture Content Visualization")
    if not inventory.empty:
        bin_fig = create_bin_visualization(bin_diameter, bin_height, inventory)
        st.plotly_chart(bin_fig)
    else:
        empty_bin_fig = create_empty_bin_visualization(bin_diameter, bin_height)
        st.plotly_chart(empty_bin_fig)
        
    # Save bins and inventory to Firebase using REST API
    bins_ref = f"{db_url}/users/{user_id}/bins.json"
    inventory_ref = f"{db_url}/users/{user_id}/{selected_bin}/inventory.json"

    if st.button("Save Changes to Firebase"):
        # Combine existing bins with newly added bins in session state
        existing_bins_response = requests.get(bins_ref)
        if existing_bins_response.status_code == 200:
            existing_bins = existing_bins_response.json() or {}
        else:
            existing_bins = {}

        updated_bins = {**existing_bins, **{bin_name: True for bin_name in st.session_state.bins}}
        response = requests.put(bins_ref, json=updated_bins)
        if response.status_code == 200:
            st.success("Bins updated successfully in Firebase.")
        else:
            st.error("Failed to update bins in Firebase.")

        # Save the current inventory of the selected bin
        inventory_data = inventory.to_dict('records')
        response = requests.put(inventory_ref, json=inventory_data)
        if response.status_code == 200:
            st.success("Inventory saved to Firebase successfully.")
        else:
            st.error("Failed to save inventory to Firebase.")

# Display messages when user ID is not entered
else:
    st.warning("Please enter a User ID to access the application.")
