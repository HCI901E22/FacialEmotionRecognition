from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
print(" ")
print("____________________________________________")
print(" ")

extractor = AutoFeatureExtractor.from_pretrained("Rajaram1996/FacialEmoRecog")

img = cv2.imread('Disgust.jpg', cv2.IMREAD_COLOR)
#img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)

img = extractor(np.array(img))

print(img)
print(type(img))

transform = transforms.ToTensor()

# Convert the image to PyTorch tensor

tensor = transform(img)

# Print the converted image tensor
print(tensor.shape)

# Define a transform to convert PIL 
# image to a Torch tensor
#transform = transforms.Compose([
#    transforms.PILToTensor()
#])

#tensor = torch.from_numpy(np.array([image])).long()

# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
#img_tensor = transform(image)

import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

from transformers import AutoFeatureExtractor, AutoModelForImageClassification

extractor = AutoFeatureExtractor.from_pretrained("Rajaram1996/FacialEmoRecog")

model = AutoModelForImageClassification.from_pretrained("Rajaram1996/FacialEmoRecog")

with torch.no_grad():
    model.eval()
    print("This is the shape of torch tensor:", tensor.shape)
    type(tensor)
    logits = model(tensor[None, ...]).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])

print(" ")
print("____________________________________________")
print(" ")
