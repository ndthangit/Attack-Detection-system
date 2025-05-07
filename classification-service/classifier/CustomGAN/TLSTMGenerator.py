import torch.nn as nn
import torch.nn.functional as F

from classifier.CustomGAN.TLSTM import TLSTM


class TLSTMGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size, dropout_rate=0.5):
        super(TLSTMGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # Encoder TLSTM
        self.encoder = TLSTM(input_size, hidden_size)
        self.encoder_dropout = nn.Dropout(dropout_rate)  # Thêm dropout sau encoder
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        # Decoder TLSTM
        self.decoder = TLSTM(latent_size, hidden_size)
        self.decoder_dropout = nn.Dropout(dropout_rate)  # Thêm dropout sau decoder
        self.decoder_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, delta_t):
        """
        x: chuỗi đầu vào (batch_size, seq_len, input_size)
        delta_t: khoảng cách thời gian (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = x.size()

        # Encoder: Mã hóa chuỗi đầu vào thành vector tiềm ẩn
        encoder_outputs, h_t, c_t = self.encoder(x, delta_t)
        h_t = self.encoder_dropout(h_t)  # Áp dụng dropout sau encoder
        latent = F.relu(self.encoder_fc(h_t))  # ReLU đã được thêm từ trước

        # Chuẩn bị đầu vào cho Decoder
        latent_repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)

        # Decoder: Tái tạo chuỗi từ vector tiềm ẩn
        decoder_outputs, _, _ = self.decoder(latent_repeated, delta_t)
        decoder_outputs = self.decoder_dropout(decoder_outputs)  # Áp dụng dropout sau decoder
        reconstructed = self.decoder_fc(decoder_outputs)  # Không áp dụng ReLU ở đầu ra cuối

        return reconstructed, latent