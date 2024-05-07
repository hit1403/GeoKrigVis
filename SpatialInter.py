#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

# Variogram model (Spherical)
def spherical_variogram(h, sill, range_param):
    if h <= range_param:
        return sill * (1.5 * (h / range_param) - 0.5 * (h / range_param) ** 3)
    else:
        return sill
    
# Kriging interpolation function
def kriging_interpolation(X, Y, Z, grid_x, grid_y, variogram_model, range_param, sill):
    z_interp = np.zeros((len(grid_x), len(grid_y)))
    for i in range(len(grid_x)):
        for j in range(len(grid_y)):
            distances = np.sqrt((X - grid_x[i])**2 + (Y - grid_y[j])**2)
            semivariance = np.var(Z) - np.mean([spherical_variogram(h, sill, range_param) for h in distances])
            weights = np.array([spherical_variogram(h, sill, range_param) for h in distances])
            z_interp[i, j] = np.dot(weights, Z) / np.sum(weights)
    return z_interp

# Function to predict Z for given coordinates (X_pred, Y_pred)
def predict_z(x_coords, y_coords, phi, X_pred, Y_pred, variogram_model, range_param, sill):
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    distances = np.sqrt((x_coords - X_pred)**2 + (y_coords - Y_pred)**2)
    print("distance: ",distances)
    semivariance = np.var(phi) - np.mean([variogram_model(h, sill, range_param) for h in distances])
    print("semivariance: ",semivariance)
    weights = np.array([variogram_model(h, sill, range_param) for h in distances])
    print("weights: ",weights)
    z_prediction = np.dot(weights, phi) / np.sum(weights)
    return z_prediction

# Function to load X, Y, and Phi data from a single file
def load_data():
    data_filename = filedialog.askopenfilename(title="Select Data File")
    with open(data_filename, 'r') as file:
        content = file.read()
    print(content)
    # Check if the content is in the expected format
    #if not content.startswith("[(") or not content.endswith(")]"):
    #    return None, None, None  # Content doesn't match the expected format
    # Remove the surrounding parentheses and split the content into individual tuples
    content = content[2:-3]
    data = [eval(f'({entry})') for entry in content.split("),(")]
    # Extract x, y, and phi values from the data
    x_coords, y_coords, phi = zip(*data)
    return x_coords, y_coords, phi

# Function to perform Kriging interpolation
def perform_kriging():
    x_coords, y_coords, phi = load_data()
    grid_x = np.linspace(0, max(x_coords) + 3, 100)
    grid_y = np.linspace(0, max(y_coords) + 3, 100)
    OK = OrdinaryKriging(
        x_coords,
        y_coords,
        phi,
        variogram_model='spherical',
        verbose=True,
        enable_plotting=True,
        nlags=10,
    )
    sill = OK.variogram_model_parameters[0]
    range_param = OK.variogram_model_parameters[1]
    z_interp = kriging_interpolation(x_coords, y_coords, phi, grid_x, grid_y, spherical_variogram, range_param, sill)
    x_pred = float(x_entry.get())
    y_pred = float(y_entry.get())
    z_prediction = predict_z(x_coords, y_coords, phi, x_pred, y_pred, spherical_variogram, range_param, sill)
    # Plot the result
    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, z_interp, cmap='viridis')
    contour = plt.contourf(grid_x, grid_y, z_interp, cmap='viridis', levels=100)
    plt.scatter(x_coords, y_coords, c='white', edgecolors='k', s=100)
    plt.scatter(x_pred, y_pred, c='red', marker='x', label=f'Predicted Z: {z_prediction:.2f}')
    cbar = plt.colorbar(contour, label='Interpolated Value', extend='both')
    # Add labels for the x and y axis
    plt.title('Kriging Interpolation')
    plt.xlabel('X Axis Label')  # Add your x-axis label here
    plt.ylabel('Y Axis Label')  # Add your y-axis label here
    plt.legend()
    plt.show()
    
def mean(data):
    if len(data) == 0:
        raise ValueError("Cannot calculate mean of an empty list")
    
    mean = sum(data) / len(data)
    return mean

def Var(data):
    if len(data) < 2:
        raise ValueError("V")
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return variance
    if len(data) < 2:
        raise ValueError("V")
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return variance

def dot(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same length for dot product")
    
    result = 0
    for i in range(len(arr1)):
        result += arr1[i] * arr2[i]
    
    return result

# Create the main window
root = tk.Tk()
root.title("Kriging Interpolation")
root.configure(bg="lightblue")  # Set the background color of the main window

# Create a frame for the input section
input_frame = tk.Frame(root, bg="lightblue")  # Set the background color of the frame
input_frame.pack(padx=20, pady=20)

# Labels and Entry Widgets
label_color = "navy"  # Define a label color
entry_bg = "lightyellow"  # Define an entry background color

x_label = tk.Label(input_frame, text="X Coordinate:", fg=label_color, bg="lightblue")
x_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

x_entry = tk.Entry(input_frame, bg=entry_bg)
x_entry.grid(row=0, column=1, padx=5, pady=5)

y_label = tk.Label(input_frame, text="Y Coordinate:", fg=label_color, bg="lightblue")
y_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

y_entry = tk.Entry(input_frame, bg=entry_bg)
y_entry.grid(row=1, column=1, padx=5, pady=5)

# Create a frame for the button
button_frame = tk.Frame(root, bg="lightblue")
button_frame.pack(padx=20, pady=10)

# Interpolation Button
button_color = "lightgreen"  # Define a button color
interpolate_button = tk.Button(button_frame, text="Perform Kriging Interpolation", command=perform_kriging, bg=button_color)
interpolate_button.pack(padx=10, pady=10)

# Start the Tkinter event loop
root.mainloop()


# In[ ]:




