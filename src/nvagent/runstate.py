"""Run directory: hidden ground truth, datasets, fits, audit log and lab notebook.

Every headless agent invocation starts its own MCP server process, so all state lives on
disk. The ground truth in ``.truth/`` is read only by the simulator and the final scoring;
no agent-facing tool returns it.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

from .safety import DEFAULT_LIMITS
from .simulator import Sample

ENV_RUN_DIR = "NVAGENT_RUN_DIR"


def _plain(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_plain), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class RunState:
    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        if not (self.dir / "run.json").exists():
            raise FileNotFoundError(f"{self.dir} is not an nvagent run directory (no run.json)")

    @classmethod
    def create(
        cls,
        run_dir,
        sample: Sample,
        goal: str = "",
        budget_points: int = 3000,
        window_mhz: tuple[float, float] = (2640.0, 3100.0),
        limits: dict | None = None,
        quality_gate: dict | None = None,
    ) -> "RunState":
        root = Path(run_dir)
        for sub in ("datasets", "fits", "plots", "state", ".truth"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        _write_json(root / ".truth" / "sample.json", sample.to_dict())
        _write_json(
            root / "run.json",
            {
                "goal": goal,
                "budget_points": int(budget_points),
                "window_mhz": [float(w) for w in window_mhz],
                "limits": limits or {},
                "quality_gate": quality_gate or {},
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        _write_json(root / "state" / "counters.json", {})
        return cls(root)

    @classmethod
    def from_env(cls) -> "RunState":
        path = os.environ.get(ENV_RUN_DIR)
        if not path:
            raise RuntimeError(f"{ENV_RUN_DIR} is not set")
        return cls(path)

    @classmethod
    def from_env_or_none(cls) -> "RunState | None":
        try:
            return cls.from_env()
        except (RuntimeError, FileNotFoundError):
            return None

    # ----------------------------------------------------------------- configuration

    @property
    def config(self) -> dict:
        return _read_json(self.dir / "run.json")

    def limits(self) -> dict:
        cfg = self.config
        return {**DEFAULT_LIMITS, **cfg.get("limits", {}), "max_total_points": cfg["budget_points"]}

    def truth(self) -> Sample:
        return Sample.from_dict(_read_json(self.dir / ".truth" / "sample.json"))

    def _next(self, key: str) -> int:
        path = self.dir / "state" / "counters.json"
        counters = _read_json(path) if path.exists() else {}
        counters[key] = counters.get(key, 0) + 1
        _write_json(path, counters)
        return counters[key]

    def measurement_rng(self) -> np.random.Generator:
        """Fresh, reproducible noise for every measurement: seeded by (sample seed, index)."""
        return np.random.default_rng([self.truth().seed, self._next("measurements")])

    # ---------------------------------------------------------------------- datasets

    def save_dataset(self, kind, mode, params, x, counts, dwell_ms, x_unit) -> dict:
        ds_id = f"ds-{self._next('datasets'):03d}"
        x = np.asarray(x, dtype=float)
        record = {
            "id": ds_id,
            "kind": kind,
            "mode": mode,
            "params": params,
            "x_unit": x_unit,
            "x_range": [float(x.min()), float(x.max())],
            "n_points": int(len(x)),
            "dwell_ms": float(dwell_ms),
            "created": time.time(),
            "x": x.tolist(),
            "counts": np.asarray(counts).astype(int).tolist(),
        }
        _write_json(self.dir / "datasets" / f"{ds_id}.json", record)
        return {k: v for k, v in record.items() if k not in ("x", "counts")}

    def dataset(self, ds_id: str) -> dict:
        path = self.dir / "datasets" / f"{ds_id}.json"
        if not path.exists():
            raise KeyError(f"no dataset {ds_id}")
        record = _read_json(path)
        record["x"], record["counts"] = np.array(record["x"]), np.array(record["counts"])
        return record

    def datasets(self) -> list[dict]:
        out = []
        for path in sorted((self.dir / "datasets").glob("ds-*.json")):
            record = _read_json(path)
            out.append({k: v for k, v in record.items() if k not in ("x", "counts")})
        return out

    def points_used(self) -> int:
        return sum(d["n_points"] for d in self.datasets())

    # -------------------------------------------------------------------------- fits

    def save_fit(self, kind: str, payload: dict) -> str:
        fit_id = f"fit-{self._next('fits'):03d}"
        _write_json(self.dir / "fits" / f"{fit_id}.json", {"id": fit_id, "kind": kind, "created": time.time(), **payload})
        _write_json(self.dir / "state" / f"latest_{kind}.json", {"id": fit_id})
        return fit_id

    def fit(self, fit_id: str) -> dict:
        path = self.dir / "fits" / f"{fit_id}.json"
        if not path.exists():
            raise KeyError(f"no fit {fit_id}")
        return _read_json(path)

    def latest_fit(self, kind: str) -> dict | None:
        path = self.dir / "state" / f"latest_{kind}.json"
        return self.fit(_read_json(path)["id"]) if path.exists() else None

    def fits(self) -> list[dict]:
        return [_read_json(p) for p in sorted((self.dir / "fits").glob("fit-*.json"))]

    def plot_path(self, fit_id: str) -> Path:
        return self.dir / "plots" / f"{fit_id}.png"

    # ------------------------------------------------------------- audit and notebook

    def audit(self, event: str, **fields) -> None:
        line = json.dumps({"t": round(time.time(), 3), "event": event, **fields}, default=_plain)
        with open(self.dir / "audit.jsonl", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def note(self, role: str, text: str) -> None:
        with open(self.dir / "notebook.md", "a", encoding="utf-8") as fh:
            fh.write(f"### {time.strftime('%H:%M:%S')} [{role}]\n{text.strip()}\n\n")
        self.audit("notebook", role=role, text=text)

    def notebook_tail(self, n: int = 5) -> list[str]:
        path = self.dir / "notebook.md"
        if not path.exists():
            return []
        entries = [e.strip() for e in path.read_text(encoding="utf-8").split("### ") if e.strip()]
        return entries[-n:]

    # --------------------------------------------------------- quality-gate bookkeeping

    def count_gate_block(self, session: str) -> int:
        return self._next("gate_blocks_" + re.sub(r"[^A-Za-z0-9_-]", "_", session))

    def record_gate(self, passed: bool, issues: list[str], session: str) -> None:
        _write_json(self.dir / "state" / "quality_gate.json", {"passed": passed, "issues": issues, "session": session, "t": time.time()})

    def gate_status(self) -> dict | None:
        path = self.dir / "state" / "quality_gate.json"
        return _read_json(path) if path.exists() else None
