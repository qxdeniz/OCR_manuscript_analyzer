import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer


from data import texts, image_paths


vectorizer = CountVectorizer()  
vectorizer.fit(texts) 


class BirchBarkDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        
        label_vector = vectorizer.transform([label]).toarray()[0]
        return image, torch.tensor(label_vector, dtype=torch.float)


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


train_dataset = BirchBarkDataset(image_paths=image_paths, labels=texts, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

import torch.nn as nn

class SimpleOCRModel(nn.Module):
    def __init__(self, output_size):
        super(SimpleOCRModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(32 * 64 * 64, output_size)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x


output_size = len(vectorizer.get_feature_names_out()) 
model = SimpleOCRModel(output_size)

criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
for epoch in range(10):  
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")


torch.save(model.state_dict(), "ocr_model.pth")
print("Обучение завершено и модель сохранена.")
