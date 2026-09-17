"""Device memory helpers (pure stdlib, armv7-safe).

Used by the config auto low-RAM profile: on phones with <=2GB total RAM the
chat screen keeps a smaller live window and the mobile save-data budgets apply
without the user opting in. Never raises; unknown platforms report 0 (no auto).
"""

from __future__ import annotations


def mem_total_mb(path: str = "/proc/meminfo") -> int:
    """Total device RAM in MB, or 0 when unknown (no /proc, parse failure)."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    kb = int(parts[1])
                    return max(0, kb // 1024)
    except Exception:
        pass
    return 0


def is_low_ram(threshold_mb: int = 2048, path: str = "/proc/meminfo") -> bool:
    """True when the device reports (0, threshold_mb] total RAM."""
    try:
        total = mem_total_mb(path=path)
    except Exception:
        return False
    return 0 < total <= threshold_mb
