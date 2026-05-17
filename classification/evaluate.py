import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from train import BrainTumorDataset, get_model

# Config
DATA_DIR = "dataset/brisc2025/classification_task/test"
MODEL_PATH = "models/tumor_classifier.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR):
        print("Dataset not found.")
        return

    dataset = BrainTumorDataset(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    num_classes = len(dataset.classes)
    model = get_model(num_classes)
    
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())


    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=dataset.classes)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nFinal Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(report)


    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Brain Tumor Classification - Confusion Matrix')
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    print("Confusion matrix saved to results/confusion_matrix.png")


    with open("results/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)

if __name__ == "__main__":
    evaluate()
