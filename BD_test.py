import requests
import json

# Firebase Database URL
db_url = "https://digitaltwin-8ae1d-default-rtdb.firebaseio.com"

def fetch_inventory_data(user_id, bin_id):
    """Fetch inventory data for a specific user and bin."""
    path = f"/users/{user_id}/{bin_id}"
    response = requests.get(f"{db_url}{path}/inventory.json")
    return response.json()

def update_inventory_data(user_id, bin_id, inventory_data):
    """Update inventory data for a specific user and bin."""
    path = f"/users/{user_id}/{bin_id}/inventory"
    response = requests.put(f"{db_url}{path}.json", data=json.dumps(inventory_data))
    return response.json()

def add_inventory_item(user_id, bin_id, new_item):
    """Add a new inventory item to a specific user and bin."""
    current_data = fetch_inventory_data(user_id, bin_id)
    if current_data is None:
        current_data = []
    current_data.append(new_item)
    return update_inventory_data(user_id, bin_id, current_data)
