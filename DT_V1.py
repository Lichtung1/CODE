import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import requests
import json
import ast

# Firebase project ID
project_id = "digitaltwin-8ae1d-default-rtdb"

# Firebase Realtime Database URL
db_url = f"https://{project_id}.firebaseio.com"

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

    # Create a heatmap based on moisture data
    moisture_heatmap = np.zeros((100, 100))

    if not inventory.empty and 'Height_m' in inventory.columns and 'Moisture_Content_percent' in inventory.columns:
        # Calculate the cumulative height of the grain layers
        grain_heights = inventory['Height_m'].cumsum()
        moisture_values = inventory['Moisture_Content_percent'].values

        for i in range(len(moisture_values)):
            if i == 0:
                moisture_heatmap[:min(int(grain_heights[i] / height * 100), 100), :] = moisture_values[i]
            else:
                start_index = min(int(grain_heights[i-1] / height * 100), 100)
                end_index = min(int(grain_heights[i] / height * 100), 100)
                moisture_heatmap[start_index:end_index, :] = moisture_values[i]

        # Set moisture content outside the grain layers to transparent
        if len(grain_heights) > 0:
            last_grain_height_index = min(int(grain_heights.iloc[-1] / height * 100), 99)
            moisture_heatmap[last_grain_height_index+1:, :] = np.nan
    else:
        st.warning("Missing 'Height_m' or 'Moisture_Content_percent' column in the inventory DataFrame.")

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
    
# Streamlit UI
st.title("Grain Storage Bin Digital Twin")

# User authentication (simplified example)
user_id = st.text_input("Enter User ID")

if user_id:
    # Retrieve bins and inventory data for the user from Firebase using REST API
    bins_ref = f"{db_url}/users/{user_id}/bins.json"
    response = requests.get(bins_ref)
    if response.status_code == 200:
        bins_data = response.json() or {}
        st.session_state.bins = list(bins_data.keys())
        
        # Retrieve inventory data for each bin
        for bin_name in st.session_state.bins:
            inventory_ref = f"{db_url}/users/{user_id}/{bin_name}.json"
            inventory_response = requests.get(inventory_ref)
            if inventory_response.status_code == 200:
                inventory_data = inventory_response.json() or []
                st.session_state[f"inventory_{bin_name}"] = pd.DataFrame(inventory_data)
            else:
                st.session_state[f"inventory_{bin_name}"] = pd.DataFrame(columns=['Date', 'Commodity', 'Mass (tonnes)', 'Test Weight (kg/m3)', 'Moisture Content (%)', 'Height (m)'])
    else:
        st.warning("Failed to retrieve bins from Firebase.")
        st.stop()  # Stop execution if bins retrieval fails
    
    if not st.session_state.bins:
        st.warning("No bins found for the user.")
        st.stop()  # Stop execution if no bins are found
    
    selected_bin = st.selectbox("Select Bin", st.session_state.bins)

    if st.button("Create New Bin"):
        new_bin_name = f"Bin {len(st.session_state.bins) + 1}"
        st.session_state.bins.append(new_bin_name)
        selected_bin = new_bin_name
        # Update bins in Firebase using REST API
        response = requests.put(bins_ref, json=st.session_state.bins)
        if response.status_code != 200:
            print("Failed to update bins in Firebase.")

    # Bin dimensions
    bin_diameter = st.number_input("Bin Diameter (m):", value=10.0)
    bin_height = st.number_input("Bin Height (m):", value=20.0)
    bin_capacity_volume = calculate_bin_capacity(bin_diameter, bin_height)

    # Initialize inventory dataframe for the selected bin
    if f"inventory_{selected_bin}" not in st.session_state:
        st.session_state[f"inventory_{selected_bin}"] = pd.DataFrame(columns=['Date', 'Commodity', 'Mass (tonnes)', 'Test Weight (kg/m3)', 'Moisture Content (%)', 'Height (m)'])

    inventory = st.session_state[f"inventory_{selected_bin}"]

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

    # Display current inventory
    st.subheader("Current Inventory")
    if not inventory.empty:
        try:
            # Access the 'inventory' column and convert the string representation to a dictionary
            inventory_data_str = inventory.iloc[0]['inventory']
            st.text("Inventory Data String:")
            st.write(inventory_data_str)  # This will display the string representation of the inventory data
            
            inventory_data = ast.literal_eval(inventory_data_str)
            st.text("Inventory Data Dictionary:")
            st.write(inventory_data)  # This will display the dictionary representation of the inventory data
            
            # Create a new DataFrame from the dictionary
            inventory_df = pd.DataFrame([inventory_data])
            st.text("Inventory DataFrame:")
            st.write(inventory_df)  # This will display the DataFrame
    
            # ... rest of the code for renaming and styling ...
            
            st.dataframe(styled_inventory)  # If styled_inventory is a Styler object, this will not work as expected
            
        except (KeyError, IndexError, SyntaxError, ValueError) as e:
            st.write("Error occurred while processing the inventory data:", e)
    else:
        st.write("No inventory data available.")
        
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

    # Save bins to Firebase using REST API
    bins_ref = f"{db_url}/users/{user_id}/bins.json"
    try:
        existing_bins_response = requests.get(bins_ref)
        existing_bins = existing_bins_response.json() if existing_bins_response.status_code == 200 else {}
        
        updated_bins = {**existing_bins, **{bin_name: True for bin_name in st.session_state.bins}}
        response = requests.put(bins_ref, json=updated_bins)
        
        if response.status_code == 200:
            print("Bins saved to Firebase successfully.")
        else:
            print("Failed to save bins to Firebase.")
    except Exception as e:
        print("Error occurred while saving bins to Firebase:")
        print(str(e))

    # Save inventory to Firebase using REST API
    inventory_ref = f"{db_url}/users/{user_id}/{selected_bin}.json"
    try:
        inventory_data = inventory.to_dict('records')
        response = requests.put(inventory_ref, json=inventory_data)
        if response.status_code == 200:
            print("Inventory saved to Firebase successfully.")
        else:
            print("Failed to save inventory to Firebase.")
    except Exception as e:
        print("Error occurred while saving inventory to Firebase:")
        print(str(e))

else:
    st.warning("Please enter a User ID to access the application.")
