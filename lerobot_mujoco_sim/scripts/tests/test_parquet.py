#!/usr/bin/env python
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
root = PROJECT_ROOT / "data" / "demo_data_so101"
paths = sorted(root.glob("data/chunk-*/episode_*.parquet"))
print("Found", len(paths), "episode files")
path = paths[0]
print("Inspecting:", path)

table = pq.read_table(path)
print("Columns:", table.column_names)

# Inspect action column
col = table["action"]
print("\n[action] column length:", len(col))
rows = col.to_pylist()[:5]
for i, r in enumerate(rows):
    arr = np.array(r)
    print(f"row {i}: type={type(r)}, shape={arr.shape}, values={np.round(arr,4)}")