import torch
import torch.nn as nn

try:
    import torchcde
except ImportError:
    raise ImportError("torchcde is required. Install with: pip install torchcde")


class CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_hidden_layers=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels

        layers = []
        in_dim = hidden_channels
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_channels))
            layers.append(nn.Tanh())
            in_dim = hidden_channels
        layers.append(nn.Linear(hidden_channels, hidden_channels * input_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        batch_dims = z.shape[:-1]
        return self.net(z).view(*batch_dims, self.hidden_channels, self.input_channels)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model

        self.hidden_channels = configs.d_model
        self.num_hidden_layers = getattr(configs, "num_layers", 2)

        self.input_embedding = nn.Linear(configs.enc_in, self.hidden_channels)

        self.cde_func = CDEFunc(
            input_channels=self.hidden_channels,
            hidden_channels=self.hidden_channels,
            num_hidden_layers=self.num_hidden_layers,
        )

        if self.task_name == "anomaly_detection":
            self.readout = nn.Linear(self.hidden_channels, self.c_out)
        elif self.task_name == "precursor":
            self.anomaly_head = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.hidden_channels // 2, 1),
            )
            self.poa_head = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.hidden_channels // 2, 1),
            )
        else:
            raise ValueError(f"task_name: {configs.task_name} not supported in NCDE")

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "anomaly_detection":
            return self._anomaly_forward(x_enc)
        elif self.task_name == "precursor":
            return self._precursor_forward(x_enc)
        else:
            raise ValueError(f"task_name: {self.task_name} not supported")

    def _anomaly_forward(self, x_enc):
        batch_size, seq_len, enc_in = x_enc.shape

        embedded = self.input_embedding(x_enc)
        embedded = embedded.transpose(1, 2)

        coeffs = torchcde.natural_cubic_spline_coeffs(embedded)
        X = torchcde.CubicSpline(coeffs)

        z0 = embedded[:, :, 0].transpose(1, 2)

        zt = torchcde.cdeint(
            X=X, func=self.cde_func, z0=z0, t=X.interval, adjoint=False
        )

        final_hidden = zt[:, :, -1]

        output = self.readout(final_hidden)
        return output

    def _precursor_forward(self, x_enc):
        batch_size, seq_len, enc_in = x_enc.shape

        embedded = self.input_embedding(x_enc)
        embedded = embedded.transpose(1, 2)

        coeffs = torchcde.natural_cubic_spline_coeffs(embedded)
        X = torchcde.CubicSpline(coeffs)

        z0 = embedded[:, :, 0].transpose(1, 2)

        zt = torchcde.cdeint(
            X=X, func=self.cde_func, z0=z0, t=X.interval, adjoint=False
        )

        final_hidden = zt[:, :, -1]

        anomaly_logits = self.anomaly_head(final_hidden)
        poa_logits = self.poa_head(final_hidden)

        return anomaly_logits, poa_logits
