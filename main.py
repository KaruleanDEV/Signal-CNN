import cv2
from wincapture import capture_win_alt

#ML
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchnn import ImageClassifier, transform, device, target_to_class

def main():
    WINDOW_NAME = "War  "
    while cv2.waitKey(1) != ord('q'):
        screenshot = capture_win_alt(WINDOW_NAME)
        #load model
        state_dict = torch.load('model.pth')

        # Load the pretrained model
        model = ImageClassifier()
        model.load_state_dict(state_dict)
        model.eval()

        # CUDA OR CPU
        model = model.to(device)

        # Define the label
        label_map = target_to_class

        #convert screenshot to tensor format
        image = Image.fromarray(screenshot)
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        #get the predictions
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

        #draw a cv2 marker around detected objects
        cv2.putText(screenshot, predicted_label_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('VIS', screenshot)

if __name__ == '__main__':
    main()