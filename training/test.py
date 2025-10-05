from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Define the same class
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

df = pd.read_csv('data/cumulative_2025.10.04_01.16.10.csv')

feature_prefixes = [
    'koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth',
    'koi_ror', 'koi_impact', 'koi_model_snr', 'koi_max_sngle_ev', 'koi_max_mult_ev',
    'koi_num_transits', 'koi_tce_plnt_num', 'koi_tce_delivname',
    'koi_fpflag_', 'koi_datalink_dvr', 'koi_datalink_dvs',
    'koi_trans_mod', 'koi_model_dof', 'koi_model_chisq',
    'koi_fwm_', 'koi_dicco_', 'koi_dikco_'
]

# Collect matching columns dynamically
feature_cols = []
for prefix in feature_prefixes:
    feature_cols.extend(df.filter(regex=f'^{prefix}').columns)

features = df[feature_cols]

labels = LabelEncoder().fit_transform(df['koi_disposition'])

features = features.select_dtypes(include=['number']).fillna(0)

koi = df['koi_score'].fillna(0).values
koi_prad = df['koi_prad'].fillna(0).values

koi_prad = koi_prad / koi_prad.max()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)
koi_tensor = torch.tensor(koi_prad, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor, koi_tensor)

model = ExoplanetClassifier(input_size=features.shape[1])
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

correct = 0
total = 0

guess_list = []

correct = 0
total = 0
mse_total = 0

testloader = DataLoader(dataset, batch_size=1, shuffle=False)

too_off = 0

with torch.no_grad():
    for inputs, labels, koi_scores in testloader:
        class_out, score_out = model(inputs)
        _, predicted = torch.max(class_out, 1)

        predicted_score = score_out.squeeze().sum().item()
        actual_score = koi_scores.squeeze().sum().item()

        print(f"Predicted:   {predicted_score * df['koi_prad'].fillna(0).values.max():.4f}", end=' ')
        print(f"Actual:   {actual_score * df['koi_prad'].fillna(0).values.max():.4f}")

        if abs(predicted_score - actual_score) > 0.1:
            too_off += 1

        total += 1
        correct += (predicted == labels).sum().item()
        mse_total += ((score_out.squeeze() - koi_scores) ** 2).sum().item()

accuracy = 100 * correct / total
mse = mse_total / total

print(f"Classification Accuracy: {accuracy:.4f}%")
print(f"koi_score MSE: {mse:.6f}")
print(too_off/total)
