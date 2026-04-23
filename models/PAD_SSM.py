import torch
import torch.nn as nn

from layers.Embed import PositionalEmbedding
from layers.MambaBlock import Mamba_TimeVariant

## Embedding layers


class TokenEmbedding_cls(nn.Module):
    """TokenEmbedding with configurable kernel size(`d_kernel`)."""

    def __init__(self, c_in, d_model, d_kernel=3):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=d_kernel,
            padding="same",
            padding_mode="replicate",
            bias=False,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding_cls(nn.Module):
    """DataEmbedding with configurable kernel size(`d_kernel`) and sequence length(`seq_len`).

    To solve the warning for EigenWorms dataset (seq_len=17984) while keeping consistency comparing with other models, we set max_len=max(5000, seq_len).
    """

    def __init__(
        self,
        c_in,
        d_model,
        embed_type="fixed",  # pyright: ignore
        freq="h",  # pyright: ignore
        dropout=0.1,
        d_kernel=3,
        seq_len=5000,
    ):
        super(DataEmbedding_cls, self).__init__()
        self.value_embedding = TokenEmbedding_cls(
            c_in=c_in, d_model=d_model, d_kernel=d_kernel
        )
        self.position_embedding = PositionalEmbedding(
            d_model=d_model, max_len=max(5000, seq_len)
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


## PAD SSM based model


class MambaPAD(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.dropout = configs.dropout
        self.num_kernels = configs.num_kernels

        # this can be changed
        self.mamba = nn.Sequential(
            Mamba_TimeVariant(
                d_model=configs.d_model,
                d_state=configs.d_ff,
                d_conv=configs.d_conv,
                expand=configs.expand,
                timevariant_dt=bool(
                    configs.tv_dt
                ),  # only available in Mamba_TimeVariant
                timevariant_B=bool(configs.tv_B),  # only available in Mamba_TimeVariant
                timevariant_C=bool(configs.tv_C),  # only available in Mamba_TimeVariant
                use_D=bool(configs.use_D),  # use D(skip connection) or not
                device=configs.device,
            ),
            nn.LayerNorm(configs.d_model),
            nn.SiLU(),  # simply choose the same activation fn as Mamba Block
        )

        if self.task_name in ["precursor"]:  # binary value per window

            self.embedding = DataEmbedding_cls(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
                configs.num_kernels,
                configs.seq_len,
            )

            self.anomaly_head = nn.Sequential(
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, 1),
            )
            self.poa_head = nn.Sequential(
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, 1),
            )
        else:
            raise ValueError(f"task_name: {configs.task_name} is not valid.")

    # basic forward for now. this one is meant for the the datasets already here
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Args:
            x_enc: (batch_size, seq_len, enc_in) - input time series
            x_mark_enc: optional, not used for PAD
            x_dec: optional, not used for PAD
            x_mark_dec: optional, not used for PAD
            mask: optional, not used for PAD

        Returns:
            anomaly_logits: (batch_size, 1) - anomaly detection logits
            poa_logits: (batch_size, 1) - precursor-of-anomaly detection logits
        """
        if self.task_name in ["precursor"]:
            embed = self.embedding(x_enc)
            mamba_out = self.mamba(embed)
            pooled = mamba_out.mean(dim=1)
            anomaly_logits = self.anomaly_head(pooled)
            poa_logits = self.poa_head(pooled)
            return anomaly_logits, poa_logits
        else:
            raise ValueError(f"task_name: {self.task_name} is not valid.")
