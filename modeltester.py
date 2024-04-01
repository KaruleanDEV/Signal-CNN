import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchnn import ImageClassifier, transform, device, target_to_class

#---------PARAM---------
directory = r'DATASET\Test_SET'
state_dict = torch.load('model.pth')

# Load the pretrained model
model = ImageClassifier()
model.load_state_dict(state_dict)
model.eval()

# CUDA OR CPU
model = model.to(device)

# Define the label ma
label_map = target_to_class

# Create a figure for plotting
fig = plt.figure(figsize=(10, 10), facecolor='lightgray')

# Loop over all files in the directory
for i, filename in enumerate(os.listdir(directory)):
    if filename.endswith(".png"):  # add more conditions if you have images of different types
        # Load the image
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path).convert("RGB")

        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Get the prediction
        with torch.no_grad():
            outputs = model(image_tensor)

        # The output has unnormalized probabilities. To get probabilities, you can run a softmax on it.
        probabilities = F.softmax(outputs, dim=1)

        # You can get the class here by finding the index with the maximum probability
        _, predicted_class = torch.max(probabilities, 1)

        # Get the confidence level
        confidence_level = torch.max(probabilities).item()

        # Get the predicted label
        predicted_label_index = predicted_class.item()
        predicted_label_name = label_map[predicted_label_index]
        if confidence_level < 0.8:
            predicted_label_name = '?'
        else:
            predicted_label_name = label_map[predicted_label_index]


        # Plot the image with the predicted label and confidence level
        ax = fig.add_subplot(8, 8, i+1)
        ax.imshow(image)
        ax.set_title(f"{predicted_label_name} \n ({confidence_level*100:.2f}%)")
        ax.axis('off')
plt.tight_layout()
plt.show()