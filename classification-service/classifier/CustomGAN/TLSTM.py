import torch
import torch.nn as nn

from classifier.CustomGAN.TLSTMCell import TLSTMCell


class TLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tlstm_cell = TLSTMCell(input_size, hidden_size)

    def forward(self, x, delta_t):
        """
        x: chuỗi đầu vào (batch_size, seq_len, input_size)
        delta_t: khoảng cách thời gian (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)
            # dt_t = delta_t[:, t, :]  # (batch_size, 1)
            # Handle both 2D and 3D delta_t
            if delta_t.dim() == 3:
                dt_t = delta_t[:, t, :]  # (batch_size, 1)
            else:
                dt_t = delta_t[:, t].unsqueeze(-1)  # (batch_size, 1)
            h_t, c_t = self.tlstm_cell(x_t, h_t, c_t, dt_t)
            outputs.append(h_t.unsqueeze(1))  # (batch_size, 1, hidden_size)

        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        return outputs, h_t, c_t