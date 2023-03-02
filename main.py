from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests
import torch
import torchvision.transforms as transforms
import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open('Angry.jpg')

extractor = AutoFeatureExtractor.from_pretrained("Rajaram1996/FacialEmoRecog")
model = AutoModelForImageClassification.from_pretrained("Rajaram1996/FacialEmoRecog")

test = extractor(image, return_tensors='pt')

with torch.no_grad():
    model.eval()
    output = model(test['pixel_values']).logits

predicted_label = output.argmax(-1).item()
print(model.config.id2label[predicted_label])