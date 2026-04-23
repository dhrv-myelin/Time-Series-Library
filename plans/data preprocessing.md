Below is a clean, implementation-grade specification. No narrative, only structured pseudocode and a PyTorch dataset aligned with your event-stream → irregular time series pipeline.

---

# 1. Transformation Pipeline — Pseudocode

## 1.1 Load + Normalize

```id="load_norm"
INPUT: csv_file

FUNCTION LOAD_DATA(csv_file):
    df ← read_csv(csv_file)

    df.timestamp ← parse_datetime(df.timestamp)
    df ← sort(df, by=["station_name", "pallet_serial_number", "timestamp"])

    RETURN df
```

---

## 1.2 Define Sequence IDs

```id="seq_id"
FUNCTION ADD_SEQUENCE_ID(df):
    FOR each row IN df:
        row.sequence_id ← (row.station_name, row.pallet_serial_number)

    RETURN df
```

---

## 1.3 Group into Sequences

```id="group_seq"
FUNCTION GROUP_SEQUENCES(df):
    sequences ← group df BY sequence_id

    RETURN sequences
```

---

## 1.4 Pivot Long → Wide (per sequence)

```id="pivot"
INPUT: sequence_df

FUNCTION PIVOT_SEQUENCE(sequence_df):

    timestamps ← unique(sequence_df.timestamp)
    timestamps ← sort(timestamps)

    metrics ← unique(sequence_df.metric_name)

    INIT:
        X ← empty matrix [T, D] filled with NaN
        M ← zeros [T, D]

    metric_index_map ← map metric_name → column_index

    FOR each row IN sequence_df:
        t_idx ← index_of(row.timestamp in timestamps)
        m_idx ← metric_index_map[row.metric_name]

        X[t_idx, m_idx] ← row.value
        M[t_idx, m_idx] ← 1

    RETURN:
        timestamps, X, M, metric_index_map
```

---

## 1.5 Compute Time Deltas

```id="delta_t"
INPUT: timestamps

FUNCTION COMPUTE_DELTA_T(timestamps):

    T ← length(timestamps)
    delta_t ← zeros[T]

    FOR t FROM 1 TO T-1:
        delta_t[t] ← (timestamps[t] - timestamps[t-1]).in_seconds()

    delta_t[0] ← 0

    RETURN delta_t
```

---

## 1.6 Time Since Last Observation (per feature)

```id="delta_obs"
INPUT: M, delta_t

FUNCTION COMPUTE_DELTA_OBS(M, delta_t):

    T, D ← shape(M)
    delta_obs ← zeros[T, D]

    FOR d FROM 0 TO D-1:
        last_time ← 0

        FOR t FROM 0 TO T-1:
            IF M[t, d] == 1:
                delta_obs[t, d] ← 0
                last_time ← 0
            ELSE:
                last_time ← last_time + delta_t[t]
                delta_obs[t, d] ← last_time

    RETURN delta_obs
```

---

## 1.7 Optional: State + Cycle Features

```id="context_features"
INPUT: sequence_df, timestamps

FUNCTION EXTRACT_CONTEXT_FEATURES(sequence_df, timestamps):

    INIT:
        state_vector[T]
        cycle_vector[T]

    FOR each timestamp t IN timestamps:

        rows ← all rows in sequence_df with timestamp == t

        state ← most_common(rows.state_context)
        cycle ← most_common(rows.cycle_count)

        state_vector[t] ← encode_state(state)
        cycle_vector[t] ← cycle

    RETURN state_vector, cycle_vector
```

---

## 1.8 Gap / Downtime Features

```id="gap_features"
INPUT: delta_t, threshold

FUNCTION COMPUTE_GAP_FEATURES(delta_t, threshold):

    T ← length(delta_t)

    is_gap ← zeros[T]
    gap_duration ← zeros[T]

    FOR t FROM 0 TO T-1:
        IF delta_t[t] > threshold:
            is_gap[t] ← 1
            gap_duration[t] ← delta_t[t]

    RETURN is_gap, gap_duration
```

---

## 1.9 Full Pipeline

```id="full_pipeline"
INPUT: csv_file

FUNCTION BUILD_DATASET(csv_file):

    df ← LOAD_DATA(csv_file)
    df ← ADD_SEQUENCE_ID(df)

    sequences ← GROUP_SEQUENCES(df)

    dataset ← empty list

    FOR each sequence_id, sequence_df IN sequences:

        timestamps, X, M, metric_map ← PIVOT_SEQUENCE(sequence_df)

        delta_t ← COMPUTE_DELTA_T(timestamps)
        delta_obs ← COMPUTE_DELTA_OBS(M, delta_t)

        state_vec, cycle_vec ← EXTRACT_CONTEXT_FEATURES(sequence_df, timestamps)

        is_gap, gap_duration ← COMPUTE_GAP_FEATURES(delta_t, threshold=GAP_T)

        dataset.append({
            "sequence_id": sequence_id,
            "timestamps": timestamps,
            "X": X,
            "M": M,
            "delta_t": delta_t,
            "delta_obs": delta_obs,
            "state": state_vec,
            "cycle": cycle_vec,
            "is_gap": is_gap,
            "gap_duration": gap_duration
        })

    RETURN dataset
```

---

# 2. PyTorch Dataset Class

This assumes you already ran the above pipeline and have `dataset` as a list of sequences.

## 2.1 Dataset Class

```python
import torch
from torch.utils.data import Dataset

class IrregularTimeSeriesDataset(Dataset):
    def __init__(self, sequences, window_size=None, horizon=1):
        """
        sequences: list of dicts from pipeline
        window_size: sliding window length (None = full sequence)
        horizon: prediction horizon
        """

        self.samples = []

        for seq in sequences:
            X = torch.tensor(seq["X"], dtype=torch.float32)
            M = torch.tensor(seq["M"], dtype=torch.float32)
            delta_t = torch.tensor(seq["delta_t"], dtype=torch.float32).unsqueeze(-1)
            delta_obs = torch.tensor(seq["delta_obs"], dtype=torch.float32)

            features = torch.cat([X, M, delta_obs, delta_t], dim=-1)

            T = features.shape[0]

            if window_size is None:
                self.samples.append((features, X))
            else:
                for t in range(T - window_size - horizon + 1):
                    x_window = features[t : t + window_size]
                    y_target = X[t + window_size : t + window_size + horizon]

                    self.samples.append((x_window, y_target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
```

---

## 2.2 Collate Function (for padding variable-length sequences)

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    xs, ys = zip(*batch)

    x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)

    lengths = torch.tensor([x.shape[0] for x in xs])

    return {
        "x": x_padded,
        "y": y_padded,
        "lengths": lengths
    }
```

---

## 2.3 Output Shape Summary

For each sample:

```
features[t] = concat(
    X[t],          # D
    M[t],          # D
    delta_obs[t],  # D
    delta_t[t]     # 1
)

→ total dimension = 3D + 1
```

---

## 3. Notes (implementation-critical)

- Do not normalize before masking is handled.
- Normalize per metric across dataset (after building all sequences).
- Keep metric index mapping consistent globally.
- Store mapping externally (needed at inference time).

---

If you want next, the logical continuation is:

- a **model skeleton (SSM + transformer variants)** that consumes exactly this tensor format
- or a **training loop with anomaly scoring (reconstruction + likelihood)**
