# -*- coding: utf-8 -*-
import argparse
import torch
import torchvision.models
import torchvision.transforms as transforms
import os
import pyheif

from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    Transform = transforms.Compose([
            transforms.Resize([224,224]),      
            transforms.ToTensor(),
            ])
    image = Transform(image)   
    image = image.unsqueeze(0)
    return image.to(device)

def predict(filename, image, model):
    image = prepare_image(image)
    with torch.no_grad():
        preds = model(image)
    print(r'%s: Popularity score: %.2f' % (filename, preds.item()))

    return preds.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict how much traction an image will get on Instagram and deletes photos with score < 3. Only work for .heic/.jpg/.jpeg/.png")
    parser.add_argument('--folder_path', type=str, default='./pics/', help='the folder that contains all the photos')
    parser.add_argument('--save_to_csv', default=False, action='store_true', help='save to CSV instead of printing on console')
    config = parser.parse_args()
    folder_path = config.folder_path

    print(config.save_to_csv)

    mapping = dict()

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith("heic") or filename.endswith("HEIC"):
            global image
            if filename.endswith("heic") or filename.endswith("HEIC"):
                heif_file = pyheif.read(os.path.join(folder_path, filename))
                image = Image.frombytes(
                            heif_file.mode, 
                            heif_file.size, 
                            heif_file.data,
                            "raw",
                            heif_file.mode,
                            heif_file.stride,
                        )
            else:
                image = Image.open(os.path.join(folder_path, filename))

            model = torchvision.models.resnet50()
            # model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
            model.fc = torch.nn.Linear(in_features=2048, out_features=1)
            model.load_state_dict(torch.load('model/model-resnet50.pth', map_location=device)) 
            model.eval().to(device)
            score = predict(filename, image, model)

            if score < 3:
                os.remove(os.path.join(folder_path, filename))
            else:
                mapping[filename] = score

    mapping = dict(sorted(mapping.items(), key=lambda item: -item[1]))

    print("---Sorted---")
    for key, value in mapping.items():
        print(key + " " + str(value))
        
    
    # print(mapping)

