import numpy as np
import matplotlib.pyplot as plt

def load_and_display_image(npy_file_path, output_image_path):
    # Load the numpy array from .npy file
    data = np.load(npy_file_path)
    
    # Display the image
    plt.imshow(data, cmap='gray')  # Use grayscale color map, or adjust as needed
    plt.colorbar()  # Optional, to show the color bar
  
    plt.show()
    
    # Save the image to a file
    plt.imsave(output_image_path, data, cmap='gray')

# Example usage
npy_file_path = './trigger/checkerboard_pattern2.npy'  # Update this to your .npy file path
output_image_path = './saved/checker_pattern.png'  # Path where you want to save the image
load_and_display_image(npy_file_path, output_image_path)
