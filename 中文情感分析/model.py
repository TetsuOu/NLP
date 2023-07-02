import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, dropout):
        super(CNN, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes])   #ModuleList将模块放入一个列表

        self.fc = nn.Linear(n_filters * len(filter_sizes), 2)

        self.dropout = nn.Dropout(dropout)  #防止过拟合

    def forward(self, text):

        # text = [batch_size, sent_len, emb_dim]

        embedded = text.unsqueeze(1)

        # embedded = [batch_size, 1, sent_len, emb_dim]

        convd = [conv(embedded).squeeze(3) for conv in self.convs]

        # conv_n = [batch_size, n_filters, sent_len - fs + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in convd]

        # pooled_n = [batch_size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))  #torch.cat使张量进行拼接

        # cat = [batch_size, n_filters * len(filter_sizes)]

        return self.fc(cat)


class lstm(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out, h_n