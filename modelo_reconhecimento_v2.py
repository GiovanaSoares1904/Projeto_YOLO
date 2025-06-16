import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import random
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import warnings
import argparse
warnings.filterwarnings('ignore')

# Configurações
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 30  # Aumentado para 30
LEARNING_RATE = 0.0001  # Reduzido para 0.00005
WEIGHT_DECAY = 1e-4
EMBEDDING_SIZE = 512
DROPOUT_RATE = 0.5
CONFIDENCE_THRESHOLD = 0.75  # Tava 0.6

# Inicializar MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Dataset personalizado
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        self.images = []
        self.labels = []
        self.label_to_idx = {"Unknown": 0}  # Classe 0 para desconhecido
        self.idx_to_label = {0: "Unknown"}
        
        # Carregar dados
        idx = 1
        for label in sorted(os.listdir(root_dir)):
            if os.path.isdir(os.path.join(root_dir, label)):
                self.label_to_idx[label] = idx
                self.idx_to_label[idx] = label
                label_dir = os.path.join(root_dir, label)
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(label_dir, img_name)
                        if cv2.imread(img_path) is not None:
                            self.images.append(img_path)
                            self.labels.append(idx)
                idx += 1
        print(f"Dataset: {len(self.images)} imagens, {len(self.label_to_idx)-1} pessoas")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Erro ao carregar {img_path}")
            return self.__getitem__(random.randint(0, len(self)-1))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"[DEBUG] Tipo após leitura: {type(image)}")
        image = self._preprocess_image(image)
        print(f"[DEBUG] Tipo após pré-processamento: {type(image)}")
        
        if self.is_training:
            image = self._apply_augmentation(image)
            print(f"[DEBUG] Tipo após augmentação: {type(image)}")
        else:
            image = Image.fromarray(image)  # Converter para PIL.Image no modo de validação
            print(f"[DEBUG] Tipo após conversão para PIL (validação): {type(image)}")
        
        if self.transform:
            image = self.transform(image)
            print(f"[DEBUG] Tipo após transformação: {type(image)}")
        
        return image, label, img_path

    def _preprocess_image(self, image):
        """Pré-processamento com MediaPipe"""
        results = face_detection.process(image)
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = image.shape[:2]
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            x, y = max(0, x-20), max(0, y-20)
            width, height = min(w-x, width+40), min(h-y, height+40)
            face = image[y:y+height, x:x+width]
            if face.size > 0:
                face = cv2.resize(face, (256, 256))
                face = cv2.bilateralFilter(face, 5, 50, 50)
                return face
        return cv2.resize(image, (256, 256))

    def _apply_augmentation(self, image):
        """Augmentação controlada"""
        pil_image = Image.fromarray(image)
        if random.random() < 0.3:
            pil_image = ImageEnhance.Brightness(pil_image).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.3:
            pil_image = ImageEnhance.Contrast(pil_image).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.5:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        return pil_image

# Modelo CNN
class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Sequential(
            nn.Linear(num_features, EMBEDDING_SIZE),
            nn.BatchNorm1d(EMBEDDING_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE)
        )
        self.classifier = nn.Linear(EMBEDDING_SIZE, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding(features)
        logits = self.classifier(embedding)
        return logits, embedding

# Função de treinamento
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total if total > 0 else 0
        val_acc = evaluate_model(model, val_loader)
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_face_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping. Melhor acurácia: {best_val_acc:.2f}%")
            break

# Função de avaliação
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(images)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0

# Função para processar imagem ou vídeo
def process_media(model, media_path, class_names, output_dir="output"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (255, 255, 0), 0: (0, 0, 255)}  # Verde, Amarelo, Ciano, Vermelho
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if media_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Processar imagem
        image = cv2.imread(media_path)
        if image is None:
            print(f"Erro ao carregar {media_path}")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = image.shape[:2]
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                x, y = max(0, x-20), max(0, y-20)
                width, height = min(w-x, width+40), min(h-y, height+40)
                face = image_rgb[y:y+height, x:x+width]
                
                if face.size > 0:
                    face = cv2.resize(face, (256, 256))
                    face_pil = Image.fromarray(face)
                    face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits, _ = model(face_tensor)
                        probabilities = torch.softmax(logits, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        label_idx = predicted.item() if confidence.item() > CONFIDENCE_THRESHOLD else 0
                        label = class_names[label_idx]
                        conf = confidence.item()
                    
                    color = colors.get(label_idx, (0, 0, 255))
                    cv2.rectangle(image, (x, y), (x+width, y+height), color, 2)
                    cv2.putText(image, f"{label} ({conf:.2f})", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        output_path = os.path.join(output_dir, os.path.basename(media_path))
        cv2.imwrite(output_path, image)
        print(f"Imagem processada salva em: {output_path}")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    elif media_path.lower().endswith(('.mp4', '.avi')):
        # Processar vídeo
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            print(f"Erro ao abrir vídeo {media_path}")
            return
        out_path = os.path.join(output_dir, f"processed_{os.path.basename(media_path)}")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                             cap.get(cv2.CAP_PROP_FPS), 
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)
                    x, y = max(0, x-20), max(0, y-20)
                    width, height = min(w-x, width+40), min(h-y, height+40)
                    face = frame_rgb[y:y+height, x:x+width]
                    
                    if face.size > 0:
                        face = cv2.resize(face, (256, 256))
                        face_pil = Image.fromarray(face)
                        face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            logits, _ = model(face_tensor)
                            probabilities = torch.softmax(logits, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                            label_idx = predicted.item() if confidence.item() > CONFIDENCE_THRESHOLD else 0
                            label = class_names[label_idx]
                            conf = confidence.item()
                        
                        color = colors.get(label_idx, (0, 0, 255))
                        cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)
                        cv2.putText(frame, f"{label} ({conf:.2f})", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        print(f"Vídeo processado salvo em: {out_path}")

# Função para processar webcam
def process_webcam(model, class_names):
    model.eval()
    cap = cv2.VideoCapture(0)  # 0 para webcam padrão
    if not cap.isOpened():
        print("Erro ao abrir a webcam")
        return
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (255, 255, 0), 0: (0, 0, 255)}  # Verde, Amarelo, Ciano, Vermelho
    
    print("Pressione 'q' para sair da webcam")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                x, y = max(0, x-20), max(0, y-20)
                width, height = min(w-x, width+40), min(h-y, height+40)
                face = frame_rgb[y:y+height, x:x+width]
                
                if face.size > 0:
                    face = cv2.resize(face, (256, 256))
                    face_pil = Image.fromarray(face)
                    face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits, _ = model(face_tensor)
                        probabilities = torch.softmax(logits, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        label_idx = predicted.item() if confidence.item() > CONFIDENCE_THRESHOLD else 0
                        label = class_names[label_idx]
                        conf = confidence.item()
                    
                    color = colors.get(label_idx, (0, 0, 255))
                    cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Função para carregar o modelo
def load_model(model_path, num_classes):
    model = FaceRecognitionModel(num_classes).to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Modelo carregado de: {model_path}")
    else:
        print(f"Modelo não encontrado em {model_path}. Treinamento necessário.")
        exit(1)
    return model

# Função principal
def main(args):
    train_dir = "dataset_train"  # Estrutura: dataset_train/Person1/, dataset_train/Person2/, dataset_train/Person3/
    
    # Transformações
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Carregar dataset para obter class_names
    dataset = FaceDataset(train_dir, transform=train_transform, is_training=True)
    if len(dataset) == 0:
        print("Dataset vazio!")
        return
    
    class_names = [dataset.idx_to_label[i] for i in range(len(dataset.label_to_idx))]
    num_classes = len(dataset.label_to_idx)

    if not args.test_only:
        # Dividir dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        val_dataset.dataset.transform = val_transform
        val_dataset.dataset.is_training = False
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        # Inicializar modelo
        model = FaceRecognitionModel(num_classes).to(DEVICE)
        # Congelar camadas iniciais
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
        for param in model.embedding.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Treinar
        print("Iniciando treinamento...")
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS)
        
        # Avaliar
        val_acc = evaluate_model(model, val_loader)
        print(f"Acurácia final de validação: {val_acc:.2f}%")
    else:
        # Carregar modelo para teste
        model = load_model("best_face_model.pt", num_classes)
    
    # Testar em imagens, vídeos ou webcam
    if args.use_webcam:
        print("Iniciando teste com webcam...")
        process_webcam(model, class_names)
    else:
        test_media = args.test_files if args.test_files else [
            r"C:\Users\vmarq\Documents\Aprendizado de Maquina\Modelo_CNN\Vanessa.mp4",
            # Adicione mais imagens aqui, por exemplo:
            r"C:\Users\vmarq\Documents\Aprendizado de Maquina\Modelo_CNN\WIN_20250614_16_48_50_Pro.jpg",
            r"C:\Users\vmarq\Documents\Aprendizado de Maquina\Modelo_CNN\WIN_20250614_16_48_48_Pro.jpg"
        ]
        for media in test_media:
            if os.path.exists(media):
                print(f"\nProcessando {media}...")
                process_media(model, media, class_names)
            else:
                print(f"Arquivo {media} não encontrado!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar ou testar modelo de reconhecimento facial")
    parser.add_argument('--test-only', action='store_true', help="Apenas testar com modelo salvo, sem treinar")
    parser.add_argument('--test-files', nargs='+', help="Lista de arquivos de teste (imagens ou vídeos)")
    parser.add_argument('--use-webcam', action='store_true', help="Usar webcam para teste em tempo real")
    args = parser.parse_args()
    main(args)