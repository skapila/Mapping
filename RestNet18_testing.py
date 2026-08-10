import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

# --- Rebuild the frozen backbone ---
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()
for param in model.parameters():
    param.requires_grad = False

# --- The preprocessing pipeline ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

resize_crop = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

to_tensor_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# --- Load a real image ---
image_path = "/home/om/Documents/NOTES/Samarth/ML/Image_classification_layer_1/Mapping/Sample_images/DJI_20260328162623_0015_Z.JPG"
image = Image.open(image_path)
if image.mode != "RGB":
    image = image.convert("RGB")

print(f"Original image size: {image.size}")

# Step A: resize + center crop only (still a normal viewable image, 0-255 pixels)
cropped_image = resize_crop(image)
cropped_image.save("step2_resized_224.jpg")
print("Saved step2_resized_224.jpg  <- what the network actually 'frames', before normalization")

# Step B: convert to tensor + normalize (this is what actually feeds the network)
tensor = to_tensor_normalize(cropped_image)
print(f"\nAfter transform, tensor shape: {tensor.shape}")
print(f"Pixel value range after normalize: min={tensor.min():.3f}, max={tensor.max():.3f}")

# Step C: denormalize back to a viewable image, to visually confirm the
# normalize step didn't corrupt anything -- this should look ~identical
# to step2_resized_224.jpg. Formula: pixel = normalized * std + mean
mean_tensor = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std_tensor = torch.tensor(IMAGENET_STD).view(3, 1, 1)
denormalized = tensor * std_tensor + mean_tensor
denormalized = denormalized.clamp(0, 1)  # guard against tiny float rounding outside [0,1]

denormalized_image = transforms.ToPILImage()(denormalized)
denormalized_image.save("step2_denormalized_preview.jpg")
print("Saved step2_denormalized_preview.jpg  <- normalized tensor, converted back for viewing")

# --- Run through the frozen backbone ---
batched = tensor.unsqueeze(0)
with torch.no_grad():
    feature = model(batched)
feature = feature.squeeze(0)

print(f"\nFeature vector shape: {feature.shape}")
print(f"First 10 values: {feature[:10]}")

