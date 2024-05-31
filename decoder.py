import torch
import torch.nn as nn
from attention import LuongAttention

class Decoder(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim, tf=False, attention_method="dot"):
        super(Decoder, self).__init__()
        self.use_tf = tf
        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim
        self.init_h = nn.Linear(encoder_dim, 512)
        self.init_c = nn.Linear(encoder_dim, 512)
        self.tanh = nn.Tanh()
        self.deep_output = nn.Linear(2048, vocabulary_size)
        self.dropout = nn.Dropout()
        self.attention = LuongAttention(encoder_dim, 512)
        self.embedding = nn.Embedding(vocabulary_size, 512)
        self.lstm = nn.LSTMCell(512 + 512, 512)

    def forward(self, img_features, captions):
        batch_size = img_features.size(0)
        h, c = self.get_init_lstm_state(img_features)
        max_timespan = max([len(caption) for caption in captions]) - 1

        prev_words = torch.zeros(batch_size, 1).long().to(img_features.device)
        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).to(img_features.device)
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).to(img_features.device)

        if self.use_tf and self.training:
            embedding = self.embedding(captions)
        else:
            embedding = self.embedding(prev_words)

        for t in range(max_timespan):
            
            if self.use_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], h), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, h), dim=1)
            
            h, c = self.lstm(lstm_input, (h, c))

            context, alpha = self.attention(img_features, h)
            output = self.deep_output(self.dropout(context))  # Generate predictions from the context vector

            preds[:, t] = output
            alphas[:, t] = alpha
            if not self.training or not self.use_tf:
                prev_words = output.argmax(dim=1, keepdim=True)

        return preds, alphas

    def caption(self, img_features, beam_size=5, max_length=20):
        batch_size = img_features.size(0)
        h, c = self.get_init_lstm_state(img_features)

        # Start tokens for each example in the batch
        prev_words = torch.zeros(batch_size, 1).long().to(img_features.device)
        captions = torch.zeros(batch_size, max_length).long().to(img_features.device)
        alphas = torch.zeros(batch_size, max_length, img_features.size(1)).to(img_features.device)

        for t in range(max_length):
            embeddings = self.embedding(prev_words).squeeze(1)
            lstm_input = torch.cat((embeddings, h), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            context, alpha = self.attention(img_features, h)
            output = self.deep_output(self.dropout(context))
            preds = output.argmax(dim=1, keepdim=True)
            captions[:, t] = preds.squeeze(1)
            prev_words = preds
            alphas[:, t, :] = alpha

        return captions, alphas

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)
        h = self.tanh(self.init_h(avg_features))
        c = self.tanh(self.init_c(avg_features))
        return h, c