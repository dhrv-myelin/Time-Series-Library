import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset


def load_data(csv_path: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path)
    df = df.drop(["pallet_serial_number", "id"])
    df = df.with_columns(
        pl.col("timestamp").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f %z")
    )
    df = df.with_columns(
        pl.concat_str(
            [
                pl.col("timestamp").dt.date().cast(pl.Utf8),
                pl.col("cycle_count").cast(pl.Utf8),
            ],
            separator="_",
        ).alias("sequence_id")
    )
    return df


def group_sequences(df: pl.DataFrame) -> list[pl.DataFrame]:
    return [group for _, group in df.group_by("sequence_id")]


def pivot_sequence(seq_df: pl.DataFrame) -> tuple:
    timestamps = seq_df["timestamp"].unique().sort().to_list()
    metrics = seq_df["metric_name"].unique().sort().to_list()

    t_map = {t: i for i, t in enumerate(timestamps)}
    m_map = {m: i for i, m in enumerate(metrics)}

    T, D = len(timestamps), len(metrics)
    X = np.full((T, D), np.nan)
    M = np.zeros((T, D))

    for row in seq_df.iter_rows(named=True):
        t_idx = t_map[row["timestamp"]]
        m_idx = m_map[row["metric_name"]]
        X[t_idx, m_idx] = row["value"]
        M[t_idx, m_idx] = 1

    timestamps_dt = seq_df["timestamp"].unique().sort()
    state_context = []
    for ts in timestamps_dt:
        rows = seq_df.filter(pl.col("timestamp") == ts)
        if len(rows) > 0:
            state_context.append(rows["state_context"][0])
        else:
            state_context.append("UNKNOWN")

    return timestamps, X, M, m_map, state_context


def compute_delta_t(timestamps) -> np.ndarray:
    T = len(timestamps)
    delta_t = np.zeros(T)
    for t in range(1, T):
        delta_t[t] = (timestamps[t] - timestamps[t - 1]).total_seconds()
    return delta_t


def compute_delta_obs(M: np.ndarray, delta_t: np.ndarray) -> np.ndarray:
    T, D = M.shape
    delta_obs = np.zeros((T, D))

    for d in range(D):
        last_time = 0
        for t in range(T):
            if M[t, d] == 1:
                delta_obs[t, d] = 0
                last_time = 0
            else:
                last_time += delta_t[t]
                delta_obs[t, d] = last_time
    return delta_obs


def compute_gap_features(delta_t: np.ndarray, threshold_seconds: float = 300) -> tuple:
    is_gap = (delta_t > threshold_seconds).astype(float)
    gap_duration = np.where(is_gap == 1, delta_t, 0)
    return is_gap, gap_duration


def build_dataset(csv_path: str, gap_threshold: float = 300):
    df = load_data(csv_path)
    sequences = group_sequences(df)
    dataset = []

    for seq_df in sequences:
        timestamps, X, M, metric_map, state_context = pivot_sequence(seq_df)
        delta_t = compute_delta_t(timestamps)
        delta_obs = compute_delta_obs(M, delta_t)
        is_gap, gap_duration = compute_gap_features(delta_t, gap_threshold)

        dataset.append(
            {
                "sequence_id": seq_df["sequence_id"][0],
                "timestamps": timestamps,
                "X": X,
                "M": M,
                "metric_names": list(metric_map.keys()),
                "state_context": state_context,
                "delta_t": delta_t,
                "delta_obs": delta_obs,
                "is_gap": is_gap,
                "gap_duration": gap_duration,
            }
        )

    return dataset


class ProcessAwareIrregularDataset(Dataset):
    def __init__(self, sequences, window_size=None, horizon=1):
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
                    self.samples.append(
                        (
                            features[t : t + window_size],
                            X[t + window_size : t + window_size + horizon],
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
