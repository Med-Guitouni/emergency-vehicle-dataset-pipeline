import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import cv2
import os

# download categories
print("Downloading Places365 categories...")
url = 'https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt'
classes = []
with urllib.request.urlopen(url) as f:
    for line in f:
        classes.append(line.decode().strip().split(' ')[0][3:])
print(f"Loaded {len(classes)} scene categories")

# load model
print("Loading ResNet18 Places365 model...")
model = models.resnet18(num_classes=365)
checkpoint = torch.hub.load_state_dict_from_url(
    'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar',
    map_location='cpu'
)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()
print("Model loaded")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# test on a few frames from the video
from preprocessor import VideoPreprocessor
video = [f for f in os.listdir('videos') if f.endswith('.mp4')][0]
p = VideoPreprocessor(f'videos/{video}')
frames = p.extract_frames(fps=1)

for t in [0, 30, 60, 90, 120, 150]:
    frame = frames[t]['frame']
    img = Image.fromarray(frame[:, :, ::-1])
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top5_probs, top5_idx = probs.topk(5)

    print(f"\nt={t}s top 5 scenes:")
    for i in range(5):
        print(f"  {classes[top5_idx[0][i]]:30s} {top5_probs[0][i].item():.3f}")