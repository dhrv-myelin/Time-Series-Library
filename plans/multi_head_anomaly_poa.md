# Dual-Head Anomaly + Precursor-of-Anomaly Model

## Objective

Implement multi-task learning:

1. anomaly detection (current window)
2. precursor-of-anomaly (future window)

Inspired by NCDE-based PAD framework :contentReference[oaicite:0]{index=0}

---

## Key Idea

Shared encoder learns representation:
h_t

Two heads:
anomaly_head(h_t)
poa_head(h_t)

---

## Inputs

window w_i:
sequence chunk [t ... t+W]

next window w\_{i+1}:
used for supervision

---

## Model

### Shared Encoder

- transformer OR SSM
- outputs:
  h_T (final state)

---

### Head 1: Anomaly Detection

y_anom = sigmoid(FC(h_T))

target:
does current window contain anomaly?

---

### Head 2: Precursor-of-Anomaly (PoA)

y_poa = sigmoid(FC(h_T))

target:
does next window contain anomaly?

---

## Training Strategy

### 1. Supervised (if labels exist)

L*anom = BCE(y_anom, label_i)
L_poa = BCE(y_poa, label*{i+1})

---

### 2. Self-Supervised (recommended)

Generate synthetic anomalies:

- time distortion
- value spikes
- missing chunks

---

### 3. Knowledge Distillation (critical)

Teacher:
anomaly*head(w*{i+1})

Student:
poa_head(w_i)

Loss:
L*kd = BCE(y_poa_i, y_anom*{i+1})

This matches PAD idea:

- future anomaly prediction via teacher signal :contentReference[oaicite:1]{index=1}

---

## Total Loss

L_total =
L_anom

- λ1 \* L_poa
- λ2 \* L_kd

---

## Training Loop

FOR each batch:
compute h*i from w_i
compute h*{i+1} from w\_{i+1}

    y_anom_i
    y_anom_{i+1}
    y_poa_i

    compute:
        L_anom
        L_poa
        L_kd

    update model

---

## Output

At inference:

- anomaly score (current)
- risk score (future anomaly likelihood)

---

## Why This Works

- anomaly head learns "what is bad"
- PoA head learns "what leads to bad"
- shared encoder improves both (multi-task effect confirmed in paper) :contentReference[oaicite:2]{index=2}

---

## Extensions

- add state-aware conditioning
- use FSM violations as additional supervision
- add uncertainty (calibration)
