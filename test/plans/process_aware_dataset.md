# Process-Aware Irregular Time Series Dataset

## Objective

Convert event-stream logs + FSM (YAML) into a hybrid dataset:

- irregular time series (X, Δt)
- process-aware structure (state, transitions, durations)

---

## Inputs

- process_metrics.csv
- process_logic.yaml

---

## Outputs

Per sequence:

- X: [T, D] # metric values
- M: [T, D] # mask
- delta_t: [T]
- state_id: [T]
- state_embedding (optional later)
- transition_id: [T]
- time_in_state: [T]
- state_duration (aggregated)
- transition_duration (aggregated)

---

## Steps

### 1. Parse FSM

- load YAML
- build:
  state_to_id
  transition_map[(state, event)] → next_state
  expected_metrics_per_state

---

### 2. Build Sequences

group by:
(station_name, pallet_serial_number)

sort by timestamp

---

### 3. Reconstruct State Timeline

FOR each sequence:
current_state ← initial_state

    FOR each row:
        event ← infer from row
        next_state ← FSM[current_state, event]

        record:
            state[t]
            transition[t]

        current_state ← next_state

---

### 4. Pivot Metrics

- pivot metric_name → columns
- build X and mask M

---

### 5. Compute Temporal Features

- delta_t[t] = t[i] - t[i-1]
- time_in_state:
  reset when state changes

---

### 6. Transition Features

FOR each state change:
transition_duration = time difference

store:
per timestep + aggregated per transition

---

### 7. State Encoding

- map state → integer
- optional:
  embedding lookup table (deferred to model)

---

### 8. Validation Checks

- invalid transitions → flag
- missing expected metrics → flag
- long state duration → flag

---

## Final Data Structure

sequence = {
"X": tensor[T, D],
"M": tensor[T, D],
"delta_t": tensor[T],
"state_id": tensor[T],
"transition_id": tensor[T],
"time_in_state": tensor[T]
}

---

## Notes

- do NOT resample time
- preserve irregularity
- FSM is source of truth for structure
