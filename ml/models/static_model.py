import torch
import torch.nn as nn
import torch.nn.functional as F

class StaticHandCNN(nn.Module):
    """
    1D CNN for static hand posture classification (Alphabet and Numbers).
    Input: 126-D feature vector (21 left hand + 21 right hand landmarks * 3 coordinates).
    """
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super(StaticHandCNN, self).__init__()
        
        # Input is reshaped to (B, 3, 42) for Conv1D where C=3, L=42
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (B, 126)
        B = x.size(0)
        
        # Reshape to (B, 42, 3) and then transpose to (B, 3, 42) for PyTorch Conv1d
        x = x.view(B, 42, 3).transpose(1, 2)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Global Average Pooling (B, 128, 42) -> (B, 128, 1) -> (B, 128)
        x = torch.mean(x, dim=2)
        
        # Classification head
        x = self.dropout2(x)
        x = self.fc(x)
        
        return x
