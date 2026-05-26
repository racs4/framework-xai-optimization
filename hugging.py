from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
from mestradolib.literature import run_literature

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/test-stuff2017/000000000001.jpg",
]
image1 = Image.open(requests.get(urls[0], stream=True).raw)
image2 = Image.open(requests.get(urls[1], stream=True).raw)

labels = ["a photo of a cat", "a photo of a dog"]

inputs = processor(
    text=labels,
    images=[image1, image2],
    return_tensors="pt",
    padding=True,
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(
    dim=1
)  # we can take the softmax to get the label probabilities

print(probs)  # prints: [[0.9927937 0.00720632]]
result = probs.argmax().item()
print(labels[result])  # prints: "a photo of a cat"
