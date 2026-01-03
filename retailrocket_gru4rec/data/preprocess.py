from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class Vocab:
    pad_id: int
    oov_id: int
    item_to_id: Dict[int, int]
    id_to_item: Dict[int, int]

    def encode(self, item_ids: List[int]) -> List[int]:
        return [self.item_to_id.get(int(x), self.oov_id) for x in item_ids]

    def decode(self, token_ids: List[int]) -> List[int]:
        return [int(self.id_to_item.get(int(x), -1)) for x in token_ids]


def unzip_raw(zip_path: Path, output_dir: Path) -> None:
    """Unzip kaggle dataset archive into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


def _build_sessions(
    events: pd.DataFrame,
    session_gap_minutes: int,
    min_session_len: int,
    max_session_len: int,
) -> pd.DataFrame:
    """Build sessions by visitorid with an inactivity threshold.

    Output columns:
      - session_id: int
      - visitorid: int
      - end_ts: int (last timestamp in session)
      - items: list[int] (raw item ids)
    """
    events = events.sort_values(["visitorid", "timestamp"])
    gap_ms = int(session_gap_minutes) * 60 * 1000

    rows: List[dict] = []
    session_id = 0

    for visitor_id, grp in tqdm(
        events.groupby("visitorid", sort=False),
        desc="Sessionizing",
        total=events["visitorid"].nunique(),
    ):
        ts = grp["timestamp"].to_numpy(dtype=np.int64)
        it = grp["itemid"].to_numpy(dtype=np.int64)

        # new session starts where gap > threshold
        gaps = np.diff(ts, prepend=ts[0])
        is_new = gaps > gap_ms

        start = 0
        for i in range(len(it)):
            if i > 0 and is_new[i]:
                seq = it[start:i].tolist()
                if len(seq) >= min_session_len:
                    rows.append(
                        {
                            "session_id": session_id,
                            "visitorid": int(visitor_id),
                            "end_ts": int(ts[i - 1]),
                            "items": seq[-max_session_len:],
                        }
                    )
                    session_id += 1
                start = i

        # last chunk
        seq = it[start:].tolist()
        if len(seq) >= min_session_len:
            rows.append(
                {
                    "session_id": session_id,
                    "visitorid": int(visitor_id),
                    "end_ts": int(ts[-1]),
                    "items": seq[-max_session_len:],
                }
            )
            session_id += 1

    return pd.DataFrame(rows)


def _temporal_split(
    sessions: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = float(train_frac) + float(val_frac) + float(test_frac)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    sessions = sessions.sort_values("end_ts").reset_index(drop=True)
    n = len(sessions)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = sessions.iloc[:n_train].copy()
    val = sessions.iloc[n_train : n_train + n_val].copy()
    test = sessions.iloc[n_train + n_val :].copy()
    return train, val, test


def _build_vocab(train_sessions: pd.DataFrame, min_item_interactions: int) -> Vocab:
    counts: Dict[int, int] = {}
    for seq in train_sessions["items"]:
        for item in seq:
            item = int(item)
            counts[item] = counts.get(item, 0) + 1

    kept = [item for item, c in counts.items() if c >= int(min_item_interactions)]
    kept.sort()

    pad_id = 0
    oov_id = 1

    item_to_id = {item: idx + 2 for idx, item in enumerate(kept)}
    id_to_item = {idx: item for item, idx in item_to_id.items()}
    id_to_item[pad_id] = -1
    id_to_item[oov_id] = -1

    return Vocab(pad_id=pad_id, oov_id=oov_id, item_to_id=item_to_id, id_to_item=id_to_item)


def _encode_sessions(sessions: pd.DataFrame, vocab: Vocab) -> pd.DataFrame:
    out = sessions.copy()
    out["items"] = out["items"].apply(vocab.encode)
    return out


def preprocess_retailrocket(
    input_dir: Path,
    output_dir: Path,
    min_session_len: int,
    max_session_len: int,
    test_size: Optional[float] = None,
    *,
    session_gap_minutes: int = 30,
    min_item_interactions: int = 5,
    max_users: Optional[int] = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Dict[str, Path]:
    """End-to-end preprocessing for RetailRocket.

    Reads:  input_dir/events.csv
    Writes: output_dir/{train,val,test}.parquet and vocab.json

    Note:
      - If test_size is provided (legacy style), we interpret it as:
        test_frac = test_size, val_frac = test_size, train_frac = 1 - 2*test_size
      - Otherwise we use train_frac/val_frac/test_frac (recommended).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = input_dir / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"events.csv not found in: {input_dir}")

    events = pd.read_csv(events_path, usecols=["timestamp", "visitorid", "event", "itemid"])
    events = events.dropna()
    events["timestamp"] = events["timestamp"].astype("int64")
    events["visitorid"] = events["visitorid"].astype("int64")
    events["itemid"] = events["itemid"].astype("int64")

    # Optional subsampling for course-scale runs
    if max_users is not None:
        # deterministic top-N by event count (stable & simple)
        user_counts = events.groupby("visitorid", sort=False).size()
        top_users = user_counts.nlargest(int(max_users)).index
        events = events[events["visitorid"].isin(top_users)]

    sessions = _build_sessions(
        events=events,
        session_gap_minutes=session_gap_minutes,
        min_session_len=int(min_session_len),
        max_session_len=int(max_session_len),
    )

    if len(sessions) == 0:
        raise RuntimeError(
            "No sessions were built. Try lowering min_session_len or increasing max_users."
        )

    # Decide split strategy
    if test_size is not None:
        ts = float(test_size)
        if ts <= 0 or ts >= 0.5:
            raise ValueError("test_size must be in (0, 0.5) when provided.")
        train_frac = 1.0 - 2.0 * ts
        val_frac = ts
        test_frac = ts

    train_sessions, val_sessions, test_sessions = _temporal_split(
        sessions=sessions,
        train_frac=float(train_frac),
        val_frac=float(val_frac),
        test_frac=float(test_frac),
    )

    vocab = _build_vocab(
        train_sessions=train_sessions, min_item_interactions=int(min_item_interactions)
    )
    train_sessions = _encode_sessions(train_sessions, vocab)
    val_sessions = _encode_sessions(val_sessions, vocab)
    test_sessions = _encode_sessions(test_sessions, vocab)

    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(
        json.dumps(
            {"pad_id": vocab.pad_id, "oov_id": vocab.oov_id, "item_to_id": vocab.item_to_id},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_sessions.to_parquet(train_path, index=False)
    val_sessions.to_parquet(val_path, index=False)
    test_sessions.to_parquet(test_path, index=False)

    return {"train": train_path, "val": val_path, "test": test_path, "vocab": vocab_path}
