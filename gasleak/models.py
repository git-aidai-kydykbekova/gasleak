import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


class LeakSNN(nn.Module):
    """Event-driven SNN: 32 spike inputs -> LIF(64) -> LIF(64) -> leaky-integrator readout (1).
    Output = readout membrane per time step (logit of 'leak present')."""

    def __init__(self, n_in=32, n_hid=64, beta=0.9, beta_out=0.97):
        super().__init__()
        sg = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(n_in, n_hid)
        self.lif1 = snn.Leaky(beta=beta, learn_beta=True, spike_grad=sg)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=True, spike_grad=sg)
        self.fc3 = nn.Linear(n_hid, 1)
        self.out = snn.Leaky(beta=beta_out, learn_beta=True, reset_mechanism="none")
        self.bias_out = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):                      # x: [B, T, n_in] (0/1)
        B, T, _ = x.shape
        m1 = self.lif1.reset_mem(); m2 = self.lif2.reset_mem(); m3 = self.out.reset_mem()
        logits, s1_all, s2_all = [], [], []
        for t in range(T):
            s1, m1 = self.lif1(self.fc1(x[:, t]), m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            _, m3 = self.out(self.fc3(s2), m3)
            logits.append(m3.squeeze(-1) + self.bias_out)
            s1_all.append(s1); s2_all.append(s2)
        return torch.stack(logits, 1), torch.stack(s1_all, 1), torch.stack(s2_all, 1)


class CausalCNN(nn.Module):
    """Conventional frame-based baseline on dense band levels (always on, every 1 ms frame)."""

    def __init__(self, n_in=32, ch=32, k=9, dil=(1, 2, 4)):
        super().__init__()
        self.k, self.dil = k, dil
        layers, c = [], n_in
        for d in dil:
            layers.append(nn.Conv1d(c, ch, k, dilation=d)); c = ch
        self.convs = nn.ModuleList(layers)
        self.head = nn.Conv1d(ch, 1, 1)

    def forward(self, x):                      # x: [B, T, n_in]
        h = x.transpose(1, 2)
        for conv, d in zip(self.convs, self.dil):
            h = F.relu(conv(F.pad(h, ((self.k - 1) * d, 0))))
        return self.head(h).squeeze(1)         # [B, T] logits

    def macs_per_step(self):
        return sum(c.in_channels * c.out_channels * c.kernel_size[0] for c in self.convs) + self.head.in_channels
