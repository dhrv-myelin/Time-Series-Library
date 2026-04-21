import torch
import torch.nn as nn


class EventEmbedding(nn.Module):
    def __init__(self, num_events, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_events, emb_dim)

    def forward(self, event_ids):
        return self.embedding(event_ids)


class DeltaTEncoder(nn.Module):
    """
    Encodes Δt into a vector (optional but useful)
    """

    def __init__(self, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, delta_t):
        # delta_t: (B, T)
        return self.mlp(delta_t.unsqueeze(-1))


class ContinuousSSM(nn.Module):
    """
    Implements:
        z_i = exp(A * Δt_i) z_{i-1} + B u_i

    Uses a diagonal A for stability + efficiency
    """

    def __init__(self, hidden_dim, input_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Diagonal A (log-param for stability)
        self.log_diag_A = nn.Parameter(torch.randn(hidden_dim))

        # Input projection
        self.B = nn.Linear(input_dim, hidden_dim)

    def forward(self, u, delta_t, z0=None):
        """
        u: (B, T, D)
        delta_t: (B, T)
        """
        B, T, _ = u.shape

        if z0 is None:
            z = torch.zeros(B, self.hidden_dim, device=u.device)
        else:
            z = z0

        outputs = []

        diag_A = -torch.exp(self.log_diag_A)  # ensures stability

        for t in range(T):
            dt = delta_t[:, t].unsqueeze(-1)  # (B, 1)

            # exp(A * Δt)
            transition = torch.exp(diag_A * dt)

            z = transition * z + self.B(u[:, t])
            outputs.append(z)

        return torch.stack(outputs, dim=1)  # (B, T, H)


class SSMPADModel(nn.Module):
    """
    Full model:
    - event embedding
    - Δt encoding
    - dual SSM branches
    - anomaly + PoA heads
    """

    def __init__(
        self,
        num_events,
        emb_dim=32,
        dt_dim=16,
        hidden_dim=64,
        share_A=True,
    ):
        super().__init__()

        self.event_emb = EventEmbedding(num_events, emb_dim)
        self.dt_encoder = DeltaTEncoder(dt_dim)

        input_dim = emb_dim + dt_dim

        # Two SSM branches
        self.ssm_anomaly = ContinuousSSM(hidden_dim, input_dim)
        self.ssm_poa = ContinuousSSM(hidden_dim, input_dim)

        # Optional parameter sharing
        if share_A:
            self.ssm_poa.log_diag_A = self.ssm_anomaly.log_diag_A

        # Heads
        self.head_anomaly = nn.Linear(hidden_dim, 1)
        self.head_poa = nn.Linear(hidden_dim, 1)

    def forward(self, event_ids, delta_t):
        """
        event_ids: (B, T)
        delta_t: (B, T)

        Returns:
            y_a: anomaly prediction
            y_p: precursor prediction
        """

        # embeddings
        e = self.event_emb(event_ids)  # (B, T, emb_dim)
        dt = self.dt_encoder(delta_t)  # (B, T, dt_dim)

        u = torch.cat([e, dt], dim=-1)  # (B, T, D)

        # SSM forward
        h = self.ssm_anomaly(u, delta_t)  # (B, T, H)
        z = self.ssm_poa(u, delta_t)  # (B, T, H)

        # take final state
        h_T = h[:, -1]
        z_T = z[:, -1]

        # heads
        y_a = torch.sigmoid(self.head_anomaly(h_T)).squeeze(-1)
        y_p = torch.sigmoid(self.head_poa(z_T)).squeeze(-1)

        return y_a, y_p
