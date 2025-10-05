import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
<<<<<<< HEAD:train.py
import numpy as np
=======
import time
import datetime

num_epochs = 500
lambda_score = 0.5  # weight for score loss
>>>>>>> master:training/train.py

df = pd.read_csv('data/cumulative_2025.10.04_01.16.10.csv')

feature_prefixes = [
    'koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth',
    'koi_ror', 'koi_impact', 'koi_model_snr', 'koi_max_sngle_ev', 'koi_max_mult_ev',
    'koi_num_transits', 'koi_tce_plnt_num', 'koi_tce_delivname',
    'koi_fpflag_', 'koi_datalink_dvr', 'koi_datalink_dvs',
    'koi_trans_mod', 'koi_model_dof', 'koi_model_chisq',
    'koi_fwm_', 'koi_dicco_', 'koi_dikco_'
]

df = df.dropna(subset=['koi_prad'])

print(df.shape)

# Collect matching columns dynamically
feature_cols = []
for prefix in feature_prefixes:
    feature_cols.extend(df.filter(regex=f'^{prefix}').columns)

features = df[feature_cols]

labels = LabelEncoder().fit_transform(df['koi_disposition'])
koi_score = df['koi_score'].fillna(0).values  # numeric target
koi_prad = df['koi_prad'].values

koi_prad = koi_prad / koi_prad.max()

features = features.select_dtypes(include=['number']).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test, koi_train, koi_test = train_test_split(
    X_scaled, labels, koi_prad, test_size=0.2, stratify=labels
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
koi_train_tensor = torch.tensor(koi_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
koi_test_tensor = torch.tensor(koi_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor, koi_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, koi_test_tensor)

class ExoplanetClassifier(nn.Module):
    def __init__(self, input_size, hidden=8192, num_classes=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # classification head
        self.class_head = nn.Linear(hidden // 2, num_classes)
        # regression head
        self.score_head = nn.Sequential(
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.shared(x)
        class_out = self.class_head(features)
        score_out = self.score_head(features)
        return class_out, score_out

input_size = features.shape[1]
net = ExoplanetClassifier(input_size=input_size)

class_counts = np.bincount(y_train)
total = class_counts.sum()
class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float32)

criterion_class = nn.CrossEntropyLoss(weight=class_weights)
criterion_score = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)

dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

<<<<<<< HEAD:train.py
num_epochs = 20
lambda_score = 0.5  # weight for score loss
=======
last_time = time.time()

times_taken = []
>>>>>>> master:training/train.py

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for i, (inputs, labels, koi_scores) in enumerate(dataloader):
        optimizer.zero_grad()

        class_out, score_out = net(inputs)
        loss_class = criterion_class(class_out, labels)
        loss_score = criterion_score(score_out, koi_scores)
        loss = loss_class + lambda_score * loss_score

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

<<<<<<< HEAD:train.py
        if((i + 1)%8 == 0):
            print(f"Epoch {epoch + 1}/{num_epochs}, {i+1}, Loss: {running_loss / 8:.6f}")
=======
        if(i == 0):
            times_taken.append(time.time() - last_time)
            print(f"Epoch {epoch + 1}/{num_epochs}, {i}, Loss: {running_loss / len(dataloader):.6f}, Estimate Time Left: {datetime.timedelta(seconds = (num_epochs - epoch + 1) * (sum(times_taken[-10:]) / 10))}, Last epoch: {time.time() - last_time:.2f} seconds")
            last_time = time.time()
>>>>>>> master:training/train.py
            running_loss = 0.0

net.eval()
correct = 0
total = 0
mse_total = 0

testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for inputs, labels, koi_scores in testloader:
        class_out, score_out = net(inputs)
        _, predicted = torch.max(class_out, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        mse_total += ((score_out.squeeze() - koi_scores) ** 2).sum().item()

accuracy = 100 * correct / total
mse = mse_total / total

print(f"Classification Accuracy: {accuracy:.4f}%")
print(f"koi_score MSE: {mse:.6f}")

<<<<<<< HEAD:train.py
with torch.no_grad():
    sample = torch.tensor(features.iloc[0].values, dtype=torch.float32).unsqueeze(0)
    class_out, score_out = net(sample)
    predicted_class = torch.argmax(class_out, dim=1).item()
    predicted_score = score_out.item()
    print(predicted_class, predicted_score)

torch.save(net.state_dict(), 'models/model.pth')
=======
torch.save(net.state_dict(), 'models/model_new.pth')
>>>>>>> master:training/train.py
