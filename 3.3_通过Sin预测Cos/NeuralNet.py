import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, 1, True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # return outs, h_state
        # 也可使用以下这样的返回值
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state
