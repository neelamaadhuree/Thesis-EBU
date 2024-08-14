import os
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Define the directory where images are stored
directory = "./Visualization/Attack Photos/"
font_path = "./Visualization/Attack Photos/DejaVuSans-Bold.ttf"

# Names of image sets and the order they should be in the grid
image_sets = ['Orig', 'Blend', 'Freq', 'Patch', 'Chkr']
labels = [
    "Original Image",
    "Blend Attack",
    "Frequency Domain Attack",
    "Patch Attack",
    "Blend Attack with Variant Pattern"
]

# Dimensions of each image assuming they are all the same size
image_width, image_height = Image.open(directory + "Orig1.png").size

# Extra space for labels on the left
label_width = 450

# Create a new empty image to hold the grid with extra space for labels
grid_width = image_width * 5 + label_width
grid_height = image_height * 5
grid_image = Image.new('RGB', (grid_width, grid_height), "white")

# Load and place each image in the grid and add labels
for i, set_name in enumerate(image_sets):
    y_position = image_height * i
    for j in range(1, 6):
        img_path = os.path.join(directory, f"{set_name}{j}.png")
        img = Image.open(img_path)
        grid_image.paste(img, (label_width + image_width * (j-1), y_position))

# Function to wrap text using the getbbox() method for size determination
def draw_text(draw, text, position, font, container_width, line_spacing=0):
    # Split the text to fit into the container
    lines = textwrap.wrap(text, width=20)  # Adjust width based on your needs
    y_text = position[1]
    for line in lines:
        draw.text((position[0], y_text), line, font=font, fill="black")
        # Get the height of the line using getbbox() and add additional line spacing
        line_height = font.getbbox(line)[3] - font.getbbox(line)[1] + line_spacing
        y_text += line_height

# Draw labels using a larger font
draw = ImageDraw.Draw(grid_image)
font_size = 45  # Adjusted size for better fit
font = ImageFont.truetype(font_path, font_size)
for i, label in enumerate(labels):
    # Calculate vertical center for each label
    lines = textwrap.wrap(label, width=20)
    total_height = sum(font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines)
    text_y_position = image_height * i + (image_height // 2) - (total_height // 2)
    draw_text(draw, label, (10, text_y_position), font, label_width, 15 if label == "Blend Attack with Variant Pattern" else 0)

# Save and show the final grid image with labels
grid_image.save(directory + "image_grid_with_labels.png")
grid_image.show()
