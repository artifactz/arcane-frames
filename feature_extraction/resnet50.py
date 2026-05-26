import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for Resnet50 inference')

# Transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
model = model.to(device)
model.eval()  # Set to evaluation mode
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
categories = torchvision.models.ResNet50_Weights.DEFAULT.meta["categories"]


def from_image(image):
    embed, prob = from_images([image])
    return embed[0], prob[0]

def from_images(images):
    input_batch = torch.stack([transform(image) for image in images]).to(device)

    with torch.no_grad():
        embeddings = feature_extractor(input_batch)
        embeddings = embeddings.view(embeddings.size(0), -1)  # Shape: (batch_size, 2048)
        outputs = model.fc(embeddings)
        probabilities = F.softmax(outputs, dim=1)

    return embeddings.cpu().numpy(), probabilities.cpu().numpy()
