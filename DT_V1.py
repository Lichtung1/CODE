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
        # Calculate the cumulative height of the grain layers
        grain_heights = inventory['Height (m)'].cumsum()
        moisture_values = inventory['Moisture Content (%)'].values

        for i in range(len(moisture_values)):
            if i == 0:
                moisture_heatmap[:min(int(grain_heights[i] / height * 100), 100), :] = moisture_values[i]
            else:
                start_index = min(int(grain_heights[i-1] / height * 100), 100)
                end_index = min(int(grain_heights[i] / height * 100), 100)
                moisture_heatmap[start_index:end_index, :] = moisture_values[i]

        # Set moisture content outside the grain layers to transparent
        moisture_heatmap[min(int(grain_heights[-1] / height * 100), 100):, :] = np.nan

    # Create the 3D figure
    fig = go.Figure(data=[
        go.Surface(x=x, y=y, z=z, surfacecolor=moisture_heatmap, colorscale='Viridis', colorbar=dict(title='Moisture Content (%)'))
    ])

    # Add a transparent outer shell to show the structure of the bin
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(128,128,128,0.2)']]))

    fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Height (m)'),
                      title='Grain Storage Bin Moisture Content')

    return fig
