````markdown
# FSM Integration into Transformer for Irregular Time Series

## Objective

Integrate process logic (FSM) into a transformer-based model to improve:

- anomaly detection
- precursor-of-anomaly prediction
- structural consistency

---

## 1. Inputs from FSM

Precompute per timestep:

- `state_id` → current FSM state
- `transition_id` → transition index
- `time_in_state` → duration spent in current state
- `is_valid_transition` → 0/1 flag

---

## 2. Model Input Construction

Base features:

- `X` (metric values)
- `M` (mask)
- `delta_t`

Add FSM features:

- `state_id`
- `time_in_state`
- `is_valid_transition`

---

## 3. Model Integration

### Embeddings

```python
state_emb = Embedding(num_states, model_dim)
```
````

---

### Input Fusion

```python
h = input_proj(X)

h = h + state_emb(state_id)
h = h + time_proj(delta_t)
h = h + time_proj(time_in_state)
```

---

### Transformer Encoder

```python
h = encoder(h)
h_last = h[:, -1]
```

---

### Heads

```python
y_anom = sigmoid(anomaly_head(h_last))
y_poa  = sigmoid(poa_head(h_last))
```

Optional:

```python
y_state = state_head(h_last)
```

---

## 4. FSM-Aware Features (Optional but useful)

Append to input:

```python
X = concat(X, is_valid_transition)
```

---

## 5. Loss Functions

### Base Losses

```python
L_anom = BCE(y_anom, y_anom_target)
L_poa  = BCE(y_poa, y_poa_target)
```

---

### FSM Constraint Loss

#### Invalid transitions

```python
L_fsm = (1 - is_valid_transition).mean()
```

---

#### Duration violations

```python
expected = expected_duration[state_id]
actual   = time_in_state

L_duration = ((actual - expected).clamp(min=0)).mean()
```

---

### Optional: State Prediction

```python
L_state = CrossEntropy(y_state, next_state_id)
```

---

### Total Loss

```python
L_total =
    L_anom
  + L_poa
  + λ1 * L_fsm
  + λ2 * L_duration
  + λ3 * L_state
```

---

## 6. Training Batch Format

```python
batch = {
    "x": X,
    "delta_t": delta_t,
    "state_id": state_id,
    "time_in_state": time_in_state,
    "y_anom": y_anom,
    "y_poa": y_poa
}
```

---

## 7. Key Design Principles

- Do not replace ML with FSM
- Use FSM as structured signal + constraint
- Preserve irregular time (no resampling)
- Keep masking intact

---

## 8. Resulting System

Model learns:

- temporal dynamics (transformer)
- process structure (FSM)
- anomaly patterns (multi-head)

---

## 9. Summary

```text
representation = model(data + FSM context)
constraints    = FSM rules
optimization   = joint loss
```

This yields:

- better generalization
- interpretable anomalies
- process-aware predictions

```

```
