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


class MissileDataset(Dataset):
  def __init__(self, X, class_label):
    self.data = torch.tensor(X, dtype=torch.float32)
    self.labels = torch.tensor(class_label, dtype=torch.long)

  def __len__(self):
    return len(self.labels)
    
  def __getitem__(self, idx):
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
  for epoch in range(num_epochs):
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
    
    if (epoch + 1) % 20 == 0 or epoch == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
      
    if val_accuracy > best_val_accuracy:
      best_val_accuracy = val_accuracy
      file_path = os.path.join(saving_path, 'best_model.pth') 
      torch.save(model.state_dict(), file_path)
  
  print(f'Best validation accuracy: {best_val_accuracy:.2f}%')

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
  print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
def inference(model, test_data, device):
  model.eval()
  test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
  
  with torch.no_grad():
    outputs = model(test_tensor)
    _, predicted = torch.max(outputs.data, 1)
  
  return predicted.cpu().numpy()

def inference_single_sample(model, sample, device):
  model.eval()
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


def main(missile_data, saving_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # Create datasets
  train_dataset = MissileDataset(missile_data['train_X'][:,:90,:], missile_data['train_class_label'])
  val_dataset = MissileDataset(missile_data['val_X_ori'][:,:90,:], missile_data['val_class_label'])
  test_dataset = MissileDataset(missile_data['test_X_ori'][:,:90,:], missile_data['test_class_label'])
  
  # Create dataloaders
  train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
  
  # Initialize model
  # input_dim = 6  # x, y, z
  
  input_dim = missile_data['train_X'].shape[2] # Modified: set to the actual dimension of the input data

  num_classes = len(np.unique(missile_data['train_class_label']))
  d_model = 64
  nhead = 4
  num_layers = 2
  dim_feedforward = 256
  
  model = TransformerClassifier(input_dim, num_classes, d_model, nhead, num_layers, dim_feedforward)
  
  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  
  # Train the model
  num_epochs = 500
  train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, saving_path)
  
  # Load best model and test
  file_path = os.path.join(saving_path, 'best_model.pth') 
  model.load_state_dict(torch.load(file_path))
  test_model(model, test_loader, criterion, device)
  
  # Perform inference on test data
  test_predictions = inference(model, missile_data['test_X_ori'], device)
  
  # Measure inference time for a single sample
  sample = missile_data['test_X_ori'][0]  # 첫 번째 테스트 샘플 사용
  avg_inference_time = measure_inference_time(model, sample, device)
  
  return model, test_predictions, avg_inference_time


# Usage
def classification(missile_data, saving_path):
  # Assuming missile_data is already loaded
  model, test_predictions, avg_inference_time = main(missile_data, saving_path)
  
  ''' 
  # Print for debugging
  print("Test Predictions Shape:", test_predictions.shape)
  print("Sample of Test Predictions:", test_predictions[:10])  # print the first 10 predictions
  '''
  
  print(f"Average Inference Time for Single Sample: {avg_inference_time*1000:.2f} ms")
  
  # Compare predicted and actual labels (optional)
  if 'test_class_label' in missile_data:
    accuracy = np.mean(test_predictions == missile_data['test_class_label'])
    print(f"Test Accuracy: {accuracy:.4f}")

  return test_predictions