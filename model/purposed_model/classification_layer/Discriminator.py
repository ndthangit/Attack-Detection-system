import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.32):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM để xử lý chuỗi
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

        # Tầng tuyến tính để dự đoán real/fake
        self.fc = nn.Linear(hidden_size, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: chuỗi đầu vào (batch_size, seq_len, input_size)
        """
        # LSTM xử lý chuỗi
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)

        # Lấy hidden state cuối cùng
        h_n = h_n[-1]  # (batch_size, hidden_size)
        h_n = self.dropout(h_n)  # Áp dụng dropout

        # Dự đoán real/fake
        out = self.fc(h_n)  # (batch_size, 1)
        # out = self.sigmoid(out)  # (batch_size, 1), giá trị từ 0 đến 1

        return out