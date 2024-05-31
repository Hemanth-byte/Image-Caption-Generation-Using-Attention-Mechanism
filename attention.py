import torch
import torch.nn as nn
import math  # Needed for scaled dot-product attention

class LuongAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim=512):
        super(LuongAttention, self).__init__()
        self.W = nn.Linear(decoder_dim, encoder_dim, bias=True)  # Add bias term
        self.softmax = nn.Softmax(dim=1)  # Softmax on the last dimension
        self.scale= math.sqrt(decoder_dim)

    def forward(self, img_features, hidden_state):
        transformed_hidden = self.W(hidden_state)  # [64, 2048]
        
        # Prepare for batch matrix multiplication
        transformed_hidden = transformed_hidden.unsqueeze(1)  # [64, 1, 2048]
        
        # Batch matrix multiplication
        scores = torch.bmm(transformed_hidden, img_features.transpose(1, 2))  # [64, 1, 49]
        scores = scores.squeeze(1) / self.scale  # [64, 49], scaled

        # Compute softmax
        alpha = self.softmax(scores)  # [64, 49]
        context = torch.bmm(alpha.unsqueeze(1), img_features).squeeze(1)  # [64, 2048]
        
        return context, alpha