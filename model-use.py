import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from nn import *


#model = SimpleOCRModel(num_classes=50)
model.load_state_dict(torch.load("ocr_model.pth"))
model.eval()  
print("Модель загружена")


char_map = {
    0: 'А', 1: 'Б', 2: 'В', 3: 'Г', 4: 'Д', 5: 'Е', 6: 'Ё', 
    7: 'Ж', 8: 'З', 9: 'И', 10: 'Й', 11: 'К', 12: 'Л', 13: 'М', 
    14: 'Н', 15: 'О', 16: 'П', 17: 'Р', 18: 'С', 19: 'Т', 
    20: 'У', 21: 'Ф', 22: 'Х', 23: 'Ц', 24: 'Ч', 25: 'Ш', 
    26: 'Щ', 27: 'Ъ', 28: 'Ы', 29: 'Ь', 30: 'Э', 31: 'Ю', 
    32: 'Я', 33: 'а', 34: 'б', 35: 'в', 36: 'г', 37: 'д', 
    38: 'е', 39: 'ё', 40: 'ж', 41: 'з', 42: 'и', 43: 'й', 
    44: 'к', 45: 'л', 46: 'м', 47: 'н', 48: 'о', 49: 'п'
}



from PIL import Image

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("L") 
    image = transform(image).unsqueeze(0)  
    return image



def predict_text(image_path, model):
    image = preprocess_image(image_path)
    with torch.no_grad(): 
        output = model(image)
    
  
    predicted_indices = torch.argmax(output, dim=1)
    


    predicted_text = ''.join([chr(idx.item()) for idx in predicted_indices])

    
    return predicted_text
    
image_path = "test.png"
text = predict_text(image_path, model)
print("Предсказанный текст:", text)
