# Solution scaffolding for 'Notatia bizantina'
# Feel free to use anything from this 

import pandas as pd
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

from typing import List, Tuple


# Dataset class
class NeumeDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, label_map: dict):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = f"{self.root_dir}/{row['Path']}"
        img = preprocess_image(cv2.imread(path))
        label = self.label_map[row['Effect']]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*12*12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.astype(np.float32) / 255.0  # normalize to [0,1]
    img = np.expand_dims(img, axis=0)     # for channel dimension
    return img


def load_data(csv_path: str, root_dir: str) -> Tuple[List, List, dict]:
    df = pd.read_csv(csv_path)
    unique_labels = sorted(df['Effect'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    return df, label_map


def train_model(train_csv, root_dir):
    df, label_map = load_data(train_csv, root_dir)
    dataset = NeumeDataset(train_csv, root_dir, label_map)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SimpleCNN(num_classes=len(label_map))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(10):
        for imgs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
    
    return model, label_map


# Preprocess and extract signs from sequence dataset
def process_sequence_image(model, label_map, path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect bounding boxes for individual neumes
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: b[0])  # left to right
    
    sequence = []
    inv_label_map = {v: k for k, v in label_map.items()}
    pitch = 0
    
    model.eval()
    for (x, y, w, h) in boxes:
        neum_img = gray[y:y+h, x:x+w]
        neum_img = cv2.resize(neum_img, (48, 48))
        neum_img = neum_img.astype(np.float32)/255.0
        neum_img = np.expand_dims(neum_img, axis=0)
        neum_img = np.expand_dims(neum_img, axis=0)
        neum_tensor = torch.tensor(neum_img, dtype=torch.float32)
        output = model(neum_tensor)
        pred_label = torch.argmax(output, dim=1).item()
        pred_effect = inv_label_map[pred_label]
        # accumulate pitch
        if pred_effect not in ['A', 'B']:
            pitch += int(pred_effect)
        sequence.append(pitch)
    return sequence


# Make predictions and output them to output.csv
def predict(model, label_map):
    results = []
    with open("dataset_eval.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            img_path = row["datapointID"]
            ans_seq = process_sequence_image(model, label_map, img_path)
            results.append({
                "subtaskID": 1,
                "datapointID": row["datapointID"],
                "answer": "|".join(map(str, ans_seq))
            })
    with open("output.csv", "w", encoding="utf-8") as file:
        file.write("subtaskID,datapointID,answer\n")
        for res in results:
            file.write(f"{res['subtaskID']},{res['datapointID']},{res['answer']}\n")

if __name__ == "__main__":
    model, label_map = train_model("dataset_train.csv", ".")
    predict(model, label_map)
    print("Done")
