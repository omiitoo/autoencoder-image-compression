import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from models.autoencoder import Autoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Autoencoder().to(device)
model.load_state_dict(torch.load("outputs/checkpoints/autoencoder.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

img = Image.open("test.jpg")
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    recon = model(img)

save_image(img, "outputs/reconstructions/original.png")
save_image(recon, "outputs/reconstructions/reconstructed.png")
