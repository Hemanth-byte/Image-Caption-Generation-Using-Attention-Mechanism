import torch
import torch.nn as nn

class AdaptiveAttention(nn.Module):
    def __init__(self, encoder_dim):
        super(AdaptiveAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = 512
        self.sentinel_affine = nn.Linear(self.hidden_dim, self.hidden_dim)  # Sentinel vector affine transformation
        self.U = nn.Linear(self.hidden_dim, self.hidden_dim)  # Hidden state to attention
        self.W = nn.Linear(encoder_dim, self.hidden_dim)  # Image features to attention
        self.v = nn.Linear(self.hidden_dim, 1)  # Attention to scalar
        # self.beta = nn.Linear(self.hidden_dim, encoder_dim)  # Gate for sentinel
        self.beta = nn.Linear(self.hidden_dim, 1) 
        self.transform_layer = nn.Linear(512, 2048)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features, hidden_state):
        #print("Hidden State ", hidden_state.shape)
        U_h = self.U(hidden_state)  # Transform hidden state
        W_s = self.W(img_features)  # Transform image features
        V_g = self.sentinel_affine(hidden_state)  # Generate sentinel vector
        #print("sentinel_affine " , self.sentinel_affine(hidden_state).shape)
        # Compute attention weights
        att = self.tanh(W_s + U_h.unsqueeze(1))
        e = self.v(att).squeeze(2)
        
        # Sentinel gating mechanism
        #print("Beta " , self.beta(hidden_state).shape)
        g = self.sigmoid(self.beta(hidden_state))
        
        # g = self.sigmoid(self.beta(hidden_state)).unsqueeze(-1)  # Ensure g is of shape [batch_size, 1] for correct broadcasting
        #print("g shape ", g.shape)
        #print("V_g shape", V_g.shape)
        sentinel = g * V_g
        #print("sentinal before unsqueeze shape ", sentinel.shape)
        sentinel = sentinel.unsqueeze(1) 
        #print("img shape", img_features.shape)
        #print("sentinal after unsqueeze shape ", sentinel.shape)

        # Concat sentinel with image features for attention computation
        sentinel_transformed = self.transform_layer(sentinel.squeeze(1))  # Transform sentinel; squeeze is used to remove the singleton dimension for the linear layer
        sentinel_transformed = sentinel_transformed.unsqueeze(1)  # Unsqueeze to add back the dimension for concatenation
        #print("sentinal after transformation ", sentinel_transformed.shape)
        sentinel = sentinel_transformed
        extended_features = torch.cat([img_features, sentinel], dim=1)
        extended_attention = torch.cat([e, torch.zeros(e.shape[0], 1).to(e.device)], dim=1)
        
        alpha = self.softmax(extended_attention)
        
        # Separate alpha for sentinel and use the rest for context calculation
        alpha_sentinel = alpha[:, -1]
        alpha = alpha[:, :-1]
        
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        alpha_sentinel = alpha_sentinel.unsqueeze(1).unsqueeze(-1)
        #print("Context shape ", context.shape)
        #print("alpha_sentinel shape", alpha_sentinel.shape)
        scaled_sentinel = alpha_sentinel * sentinel  # Resulting shape [64, 1, 2048]
        scaled_sentinel = scaled_sentinel.squeeze(1)  
        context += scaled_sentinel  # Add sentinel-based context

        return context, alpha

