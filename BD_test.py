import streamlit as st
import requests
import json

# Firebase Database URL
db_url = "https://digitaltwin-8ae1d-default-rtdb.firebaseio.com"

# Function to fetch inventory data
def fetch_inventory_data(user_id, bin_id):
    """Fetch inventory data for a specific user and bin."""
    path = f"/users/{user_id}/{bin_id}/inventory"
    response = requests.get(f"{db_url}{path}.json")
    if response.ok:
        data = response.json()
        if isinstance(data, list):
            # If the data is directly a list of inventory items
            return data
        elif isinstance(data, dict):
            # Handle cases where data might be a dictionary containing lists or other structures
            # Let's assume we might encounter dictionaries where you need to extract lists
            return [value for key, value in data.items() if isinstance(value, list)]
        else:
            # Return an empty list if data is not as expected
            return []
    return None

# Function to update inventory data
def update_inventory_data(user_id, bin_id, inventory_data):
    """Update inventory data for a specific user and bin."""
    path = f"/users/{user_id}/{bin_id}"
    formatted_data = [{'inventory': item} for item in inventory_data]
    response = requests.put(f"{db_url}{path}.json", data=json.dumps(formatted_data))
    return response.json()

# Streamlit interface
st.title("Grain Storage Bin Inventory Management")

# User and bin selection
user_id = st.sidebar.selectbox("Select User ID", ["User1", "User2"])
bin_id = st.sidebar.selectbox("Select Bin ID", ["Bin1", "Bin2", "Bin3", "Bin4"])

# Display current inventory
if st.sidebar.button("Fetch Inventory"):
    inventory_data = fetch_inventory_data(user_id, bin_id)
    if inventory_data:
        for item in inventory_data:
            st.write(item)
    else:
        st.write("No data found.")

# Form to add or update inventory data
with st.form("inventory_form"):
    st.write("Add/Update Inventory Item")
    commodity = st.text_input("Commodity")
    date = st.date_input("Date")
    height_m = st.number_input("Height (m)", format="%.2f")
    mass_tonnes = st.number_input("Mass (tonnes)", format="%.2f")
    moisture_content_percent = st.number_input("Moisture Content (%)", format="%.2f")
    test_weight_kg_m3 = st.number_input("Test Weight (kg/m3)", format="%.2f")

    submitted = st.form_submit_button("Submit")
    if submitted:
        new_item = {
            "Commodity": commodity,
            "Date": str(date),
            "Height_m": height_m,
            "Mass_tonnes": mass_tonnes,
            "Moisture_Content_percent": moisture_content_percent,
            "Test_Weight_kg_m3": test_weight_kg_m3
        }
        # Fetch current data and update
        current_data = fetch_inventory_data(user_id, bin_id)
        if not current_data:
            current_data = []
        current_data.append(new_item)
        update_result = update_inventory_data(user_id, bin_id, current_data)
        st.success("Inventory updated.")
        st.write(update_result)
