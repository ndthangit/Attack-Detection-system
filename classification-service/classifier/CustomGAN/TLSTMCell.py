import torch
import torch.nn as nn

class TLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Các cổng của TLSTM: input, forget, cell, output
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)  # Input gate
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)  # Forget gate
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)  # Cell gate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # Output gate

        # Tham số để xử lý thời gian (time decay)
        self.W_t = nn.Linear(1, hidden_size, bias=False)  # Time projection

    def forward(self, x, h_prev, c_prev, delta_t):
        """
        x: input tại thời điểm hiện tại (batch_size, input_size)
        h_prev: hidden state trước đó (batch_size, hidden_size)
        c_prev: cell state trước đó (batch_size, hidden_size)
        delta_t: khoảng cách thời gian (batch_size, 1)
        """
        # Kết hợp input và hidden state trước đó
        combined = torch.cat((x, h_prev), dim=1)

        # Input gate
        i_t = torch.sigmoid(self.W_i(combined))

        # Forget gate, điều chỉnh bởi thời gian
        f_t_base = torch.sigmoid(self.W_f(combined))
        # Tính time decay dựa trên delta_t
        time_decay = torch.tanh(self.W_t(delta_t))  # (batch_size, hidden_size)
        f_t = f_t_base * torch.sigmoid(time_decay)  # Kết hợp time decay vào forget gate

        # Cell gate
        c_tilde = torch.tanh(self.W_c(combined))

        # Output gate
        o_t = torch.sigmoid(self.W_o(combined))

        # Cập nhật cell state
        c_t = f_t * c_prev + i_t * c_tilde

        # Cập nhật hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t