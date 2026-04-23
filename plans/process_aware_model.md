# Process-Aware Time Series Model (SSM / Transformer Hybrid)

## Objective

Build a model that jointly uses:

- temporal signals (X, Δt)
- process signals (state, transitions)

---

## Input Per Timestep

features[t] =
concat(
X[t], # D
M[t], # D
delta_t[t], # 1
time_in_state[t], # 1
state_embedding[t] # E
)

---

## Architecture Options

### Option A: Transformer

1. Input Projection
   linear(features → hidden_dim)

2. Time Encoding
   encode(delta_t)

3. Add State Embedding
   embedding(state_id)

4. Transformer Encoder
   N layers

5. Output Heads:
   - reconstruction head
   - next-step prediction head

---

### Option B: SSM (preferred for irregular)

1. Input Projection

2. Continuous-time handling:
   - inject delta_t into state updates

3. SSM block (S4 / Mamba-style)

4. Residual stack

5. Output heads

---

## State Integration (Critical)

state_embedding = Embedding(num_states, E)

Add:
h_t = h_t + state_embedding

---

## Transition Awareness (Optional)

Add:
transition_embedding

or:
gating:
h_t = h_t \* f(transition_id)

---

## Outputs

### Reconstruction

X_hat[t]

### Forecast

X_pred[t+1]

---

## Losses

L_recon = MSE(X_hat, X)
L_forecast = MSE(X_pred, X_shifted)

---

## Anomaly Score

score[t] =
α \* reconstruction_error

- β \* forecast_error

---

## Notes

- normalize X globally
- mask-aware loss (ignore missing)
- batch by padded sequences
