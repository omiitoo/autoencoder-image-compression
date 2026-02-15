from torchvision import transforms, datasets

def get_dataset(path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(path, transform=transform)
    return dataset
