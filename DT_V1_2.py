import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import requests
import json

# Firebase Database URL
db_url = "https://digitaltwin-8ae1d-default-rtdb.firebaseio.com/"

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
        custom_colourscale = [[0.0, 'rgba(128,128,128,1)'], [9/30, 'green'], [14/30, 'yellow'], [20/30, 'red'], [1.0, 'red']]
        fig = go.Figure(data=go.Surface(x=x, y=y, z=z, surfacecolour=moisture_heatmap, colourscale=custom_colourscale, cmin=0, cmax=30, colourbar=dict(title='Moisture Content (%)')))
        fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, colourscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(128,128,128,0.2)']]))
        fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Height (m)'), title='Grain Storage Bin Moisture Content')
        return fig
    return None

def unload_grain(user_id, bin_id, mass_to_unload, bin_diameter): #unload grain from specific bin
    inventory_df = fetch_inventory_data(user_id, bin_id)
    total_mass = inventory_df['Mass_tonnes'].sum()
    if mass_to_unload > total_mass:
        st.warning(f"Not enough grain to unload. Short by {mass_to_unload - total_mass:.2f} tonnes.")
        return

    remaining_mass = mass_to_unload
    preserved_rows = []

    for index, row in inventory_df.iterrows():
        if remaining_mass <= 0:
            # If no more mass needs to be unloaded, copy the remaining rows as they are
            preserved_rows.append(row)
            continue

        # Check if the row has enough mass to unload
        if remaining_mass >= row['Mass_tonnes']:
            # Subtract the mass from remaining_mass
            remaining_mass -= row['Mass_tonnes']
            # Do not append rows where the entire mass has been unloaded
            continue  # Skip to the next row
        else:
            # Partially unload grain from this layer
            row['Mass_tonnes'] -= remaining_mass
            row['Height_m'] = row['Mass_tonnes'] * 1000 / (row['Test_Weight_kg_m3'] * np.pi * (bin_diameter / 2) ** 2)
            preserved_rows.append(row)
            remaining_mass = 0  # All required mass has been unloaded

    # Reconstruct the DataFrame from the preserved_rows list
    new_inventory = pd.DataFrame(preserved_rows)
    update_inventory_data(user_id, bin_id, new_inventory.to_dict(orient='records'))

#Stream Lit GUI
st.title("Grain Storage Bin Digital Twin")

user_id = st.sidebar.selectbox("Select User ID", ["User1", "User2"])
selected_bin = st.sidebar.selectbox("Select Bin ID", ["Bin1", "Bin2", "Bin3", "Bin4"])

# Initialize variables before they are used in any function
bin_diameter = st.number_input("Bin Diameter (m):", value=10.0, min_value=0.1, step=0.1)
bin_height = st.number_input("Bin Height (m):", value=20.0, min_value=0.1, step=0.1)

# Fetch the current inventory so it is available
inventory_df = fetch_inventory_data(user_id, selected_bin)
if inventory_df.empty:
    inventory_df = pd.DataFrame(columns=['Date', 'Commodity', 'Mass_tonnes', 'Test_Weight_kg_m3', 'Moisture_Content_percent', 'Height_m'])

# Grain input form
with st.form("inventory_form"):
    st.subheader("Add Grain to Inventory")
    commodity = st.selectbox("Commodity", ["Wheat", "Corn", "Oats", "Barley", "Canola", "Soybeans", "Rye"])
    mass = st.number_input("Mass (tonnes):", min_value=0.0, format='%.2f')
    test_weight = st.number_input("Test Weight (kg/m³):", min_value=0.0, format='%.2f')
    moisture_content = st.number_input("Moisture Content (%):", min_value=0.0, format='%.2f')
    submit_button = st.form_submit_button(label='Add Grain')

    if submit_button:
        if test_weight > 0:
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
            st.success("Inventory updated successfully.")
        else:
            st.error("Test weight must be greater than zero to calculate the height.")

# Unload grain section
st.sidebar.subheader("Unload Grain")
unload_mass = st.sidebar.number_input("Mass to unload (tonnes):", min_value=0.0, format='%.2f')
unload_button = st.sidebar.button("Unload Grain")

if unload_button:
    unload_grain(user_id, selected_bin, unload_mass, bin_diameter)
    inventory_df = fetch_inventory_data(user_id, selected_bin)  # Fetch updated inventory data after unloading

# Display the current and possibly updated inventory
st.subheader("Current Inventory")
st.write(inventory_df)

# Display the bin visualization
st.subheader("Bin Visualization")
if not inventory_df.empty:
    bin_fig = create_bin_visualization(bin_diameter, bin_height, inventory_df)
    st.plotly_chart(bin_fig)

# Future state section as a placeholder
st.subheader("Potential Future State")
st.write("This section will display the potential future state of the grain storage bin based on historical data and predictive models.")
