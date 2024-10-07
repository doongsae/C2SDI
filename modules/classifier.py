###############################################################
#                       classifier.py                         #
#         Predict binary class label using transformer        #
###############################################################

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
from tqdm import tqdm

class MissileDataset(Dataset):
  def __init__(self, X, class_label):
    # Replace NaN values with 0 or another appropriate value
    X = np.nan_to_num(X, nan=0.0)
    self.data = torch.tensor(X, dtype=torch.float32)
    self.labels = torch.tensor(class_label, dtype=torch.long)

  def __len__(self):
    return len(self.labels)
    
  def __getitem__(self, idx):
    if idx >= len(self.data):
        print(f"Invalid index {idx}, dataset size is {len(self.data)}")
    return self.data[idx], self.labels[idx]

class TransformerClassifier(nn.Module):
  def __init__(self, input_dim, num_classes, d_model, nhead, num_layers, dim_feedforward):
    super().__init__()
    self.embedding = nn.Linear(input_dim, d_model)
    self.pos_encoder = PositionalEncoding(d_model)
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    self.fc = nn.Linear(d_model, num_classes)
    
  def forward(self, x):
    x = self.embedding(x)
    x = self.pos_encoder(x)
    x = self.transformer_encoder(x)
    x = x.mean(dim=1)  # Global average pooling
    x = self.fc(x)
    return x

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=5000):
    super().__init__()
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0)]
    return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, saving_path):
  model.to(device)
  best_val_accuracy = 0
  for epoch in tqdm(range(num_epochs), desc = "Training Pullup Classifier"):
    model.train()
    train_loss = 0
    for batch_data, batch_labels in train_loader:
      batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
      optimizer.zero_grad()
      outputs = model(batch_data)
      loss = criterion(outputs, batch_labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
      
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_data, batch_labels in val_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
      
    if val_accuracy > best_val_accuracy:
      best_val_accuracy = val_accuracy
      file_path = os.path.join(saving_path, 'classifier_best_model.pth') 
      torch.save(model.state_dict(), file_path)
  
  print(f'Pullup classifier Best validation accuracy: {best_val_accuracy:.2f}%')

def test_model(model, test_loader, criterion, device):
  model.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_data, batch_labels in test_loader:
      batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
      outputs = model(batch_data)
      loss = criterion(outputs, batch_labels)
      test_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += batch_labels.size(0)
      correct += (predicted == batch_labels).sum().item()
  
  test_loss /= len(test_loader)
  test_accuracy = 100 * correct / total
  print(f'Pullup classifier Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


def inference(model, test_data, device):
  model.eval()
  # Replace NaN values before inference
  test_data = np.nan_to_num(test_data, nan=0.0)
  test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
  
  with torch.no_grad():
    outputs = model(test_tensor)
    _, predicted = torch.max(outputs.data, 1)
  
  return predicted.cpu().numpy()

def inference_single_sample(model, sample, device):
  model.eval()
  # Replace NaN values before inference
  sample = np.nan_to_num(sample, nan=0.0)
  sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
  
  with torch.no_grad():
    output = model(sample_tensor)
    _, predicted = torch.max(output.data, 1)
  
  return predicted.item()

def measure_inference_time(model, sample, device, num_runs=100):
  # Warm-up run
  _ = inference_single_sample(model, sample, device)
  
  # Measure time
  start_time = time.time()
  for _ in range(num_runs):
    _ = inference_single_sample(model, sample, device)
  end_time = time.time()
  
  avg_time = (end_time - start_time) / num_runs
  return avg_time


def main(missile_data, saving_path, num_epochs, inference_mode, existed_model_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # Replace NaN values in the input data
  for key in ['train_X_classifier', 'val_X_classifier', 'test_X_classifier']:
    missile_data[key] = np.nan_to_num(missile_data[key], nan=0.0)

  train_dataset = MissileDataset(missile_data['train_X_classifier'], missile_data['train_class_label_classifier'])
  val_dataset = MissileDataset(missile_data['val_X_classifier'], missile_data['val_class_label_classifier'])
  test_dataset = MissileDataset(missile_data['test_X_classifier'], missile_data['test_class_label'])

  train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
  
  # Initialize model. input_dim = 6
  input_dim = missile_data['train_X_classifier'].shape[2] 

  num_classes = len(np.unique(missile_data['train_class_label_classifier']))
  d_model = 64
  nhead = 4
  num_layers = 2
  dim_feedforward = 256
  
  model = TransformerClassifier(input_dim, num_classes, d_model, nhead, num_layers, dim_feedforward)
  
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  

  if not inference_mode:
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, saving_path)
  
    file_path = os.path.join(saving_path, 'classifier_best_model.pth') 
    model.load_state_dict(torch.load(file_path))

  else:
    model.load_state_dict(torch.load(existed_model_path))
    model.to(device)

  test_model(model, test_loader, criterion, device)
  test_predictions = inference(model, missile_data['test_X_classifier'], device)
  
  # Measure inference time for a single sample
  sample = missile_data['test_X_classifier'][0]  # use the first sample
  avg_inference_time = measure_inference_time(model, sample, device)
  
  return model, test_predictions, avg_inference_time


def classification(missile_data, saving_path, num_epochs, inference_mode, existed_model_path):
  # Assuming missile_data is already loaded
  model, test_predictions, avg_inference_time = main(missile_data, saving_path, num_epochs, inference_mode, existed_model_path)
  
  print(f"Average Inference Time for Single Sample: {avg_inference_time*1000:.2f} ms")

  return test_predictions