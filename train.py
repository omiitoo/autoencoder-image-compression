import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.dataset import get_dataset
from models.autoencoder import Autoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = get_dataset("data/raw")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 20

for epoch in range(epochs):
    for img, _ in loader:
        img = img.to(device)

        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "outputs/checkpoints/autoencoder.pth")
