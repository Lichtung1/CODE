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
        return pd.DataFrame(data) if data else pd.DataFrame()
    else:
        st.error(f"Failed to fetch data: {response.text}")
        return pd.DataFrame()

def update_inventory_data(user_id, bin_id, inventory_data):
    """Update inventory data for a specific user and bin."""
    response = requests.put(f"{db_url}/users/{user_id}/{bin_id}/inventory.json", json.dumps(inventory_data))
    if not response.ok:
        st.error("Failed to update data: " + response.text)

def create_bin_visualization(diameter, height, inventory):
    """ Visualization logic here """
    return go.Figure()

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
    commodity = st.selectbox("Commodity", ["Wheat", "Corn", "Oats", "Barley", "Canola", "Soybeans", "Rye"])
    mass = st.number_input("Mass (tonnes):")
    test_weight = st.number_input("Test Weight (kg/m³):")
    moisture_content = st.number_input("Moisture Content (%):")
    submit_button = st.form_submit_button("Add Grain")

    if submit_button:
        new_grain_data = pd.DataFrame([{
            'Date': datetime.date.today().isoformat(),
            'Commodity': commodity,
            'Mass_tonnes': mass,
            'Test_Weight_kg_m3': test_weight,
            'Moisture_Content_percent': moisture_content,
            'Height_m': mass * 1000 / (np.pi * (bin_diameter / 2) ** 2 * test_weight)
        }])
        inventory_df = pd.concat([inventory_df, new_grain_data], ignore_index=True)
        update_inventory_data(user_id, selected_bin, inventory_df.to_dict(orient='records'))
        st.success("Inventory updated successfully.")

st.subheader("Updated Inventory")
st.write(inventory_df)
