
# Transformer-based PAD (Irregular Time Series) — Implementation

<!--toc:start-->
- [Transformer-based PAD (Irregular Time Series) — Implementation](#transformer-based-pad-irregular-time-series-implementation)
  - [Overview](#overview)
  - [Data Preparation](#data-preparation)
    - [Input Format](#input-format)
    - [Preprocessing](#preprocessing)
  - [Time Encoding](#time-encoding)
  - [Input Embedding](#input-embedding)
  - [Transformer Encoder](#transformer-encoder)
  - [Sequence Pooling](#sequence-pooling)
  - [Model Definition](#model-definition)
  - [Forward Pass](#forward-pass)
  - [Loss Function](#loss-function)
  - [Training Loop](#training-loop)
  - [Causal Attention (Optional)](#causal-attention-optional)
  - [Inference](#inference)
  - [End](#end)
<!--toc:end-->

## Overview

This implementation defines a Transformer-based model for:

- Anomaly detection (current window)
- Precursor-of-anomaly (next window)

Key properties:

- Handles irregular time series (no interpolation)
- Uses Δt (time gaps) explicitly
- Window-based training
- Dual-task learning with optional knowledge distillation

---

## Data Preparation

### Input Format

Each raw datapoint:

```
(t_i, x_i)
```

### Preprocessing

```
FUNCTION PREPROCESS_TRANSFORMER(D):

    SORT D by timestamp

    FOR i in 1...T:
        Δt_i = t_i - t_{i-1}

    FOR each i:
        x'_i = CONCAT(x_i, Δt_i)

    SPLIT into windows:
        FOR i in range(0, T, window_size):
            w_i = [x'_i, ..., x'_{i+k}]

    RETURN {w_1, ..., w_n}
```

---

## Time Encoding

```
FUNCTION TIME_ENCODING(t_i, Δt_i):

    e_abs   = SINUSOIDAL_ENCODING(t_i)
    e_delta = MLP(Δt_i)

    RETURN e_abs + e_delta
```

---

## Input Embedding

```
FUNCTION EMBED(w):

    FOR each timestep i in window:

        v_i = LINEAR(x'_i)
        e_i = TIME_ENCODING(t_i, Δt_i)

        token_i = v_i + e_i

    RETURN sequence {token_1, ..., token_k}
```

---

## Transformer Encoder

```
FUNCTION TRANSFORMER_ENCODER(tokens, θ):

    h = tokens

    FOR layer in 1...L:

        h_attn = MHA(h, h, h; θ)
        h = LAYER_NORM(h + h_attn)

        h_ff = FFN(h; θ)
        h = LAYER_NORM(h + h_ff)

    RETURN h
```

---

## Sequence Pooling

```
FUNCTION POOL(h):

    RETURN h[CLS]
```

---

## Model Definition

```
PARAMETERS:
    θ_f   # anomaly transformer
    θ_g   # PoA transformer
    θ_c   # shared parameters (optional)
    θ_a   # anomaly head
    θ_p   # PoA head
```

```
FUNCTION TRANSFORMER_FORWARD(w, θ):

    tokens = EMBED(w)

    h_seq = TRANSFORMER_ENCODER(tokens; θ)

    h = POOL(h_seq)

    RETURN h
```

---

## Forward Pass

```
FUNCTION FORWARD_TRANSFORMER(w_i, w_{i+1}):

    # Anomaly branch
    h_i     = TRANSFORMER_FORWARD(w_i, θ_f, θ_c)
    h_i+1   = TRANSFORMER_FORWARD(w_{i+1}, θ_f, θ_c)

    # PoA branch
    z_i     = TRANSFORMER_FORWARD(w_i, θ_g, θ_c)

    # Predictions
    ŷ_a_i     = SIGMOID(FC(h_i; θ_a))
    ŷ_a_i+1   = SIGMOID(FC(h_i+1; θ_a))
    ŷ_p_i+1   = SIGMOID(FC(z_i; θ_p))

    RETURN ŷ_a_i, ŷ_a_i+1, ŷ_p_i+1
```

---

## Loss Function

```
FUNCTION LOSS(ŷ_a_i, ŷ_a_i+1, ŷ_p_i+1, y_i):

    L_a  = CE(ŷ_a_i, y_i)
    L_kd = CE(ŷ_a_i+1, ŷ_p_i+1)

    RETURN L_a + L_kd
```

---

## Training Loop

```
INITIALIZE θ_f, θ_g, θ_c, θ_a, θ_p

FOR epoch in 1...N:

    FOR each (w_i, w_{i+1}):

        ŷ_a_i, ŷ_a_i+1, ŷ_p_i+1 = FORWARD_TRANSFORMER(w_i, w_{i+1})

        L_a  = CE(ŷ_a_i, y_i)
        L_kd = CE(ŷ_a_i+1, ŷ_p_i+1)

        UPDATE θ_f, θ_a using L_a
        UPDATE θ_g, θ_p using L_kd
        UPDATE θ_c using (L_a + L_kd)
```

---

## Causal Attention (Optional)

```
FUNCTION APPLY_CAUSAL_MASK(sequence_length):

    FOR i in 1...k:
        ALLOW attention only to tokens ≤ i

    APPLY mask in MHA
```

---

## Inference

```
FUNCTION INFER(w_i):

    h_i = TRANSFORMER_FORWARD(w_i, θ_f)

    anomaly_score = SIGMOID(FC(h_i; θ_a))
    poa_score     = SIGMOID(FC(h_i; θ_p))

    RETURN anomaly_score, poa_score
```

---

## End
