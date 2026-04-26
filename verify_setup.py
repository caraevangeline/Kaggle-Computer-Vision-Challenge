#!/usr/bin/env python3
"""Pre-flight environment check. Run this before register_tables.py.

Verifies Python version, required packages, GPU availability, and data layout.
Catches the most common setup problems before they waste training time.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

KIT_DIR: Path = Path(__file__).resolve().parent

_PASS = "[OK]"
_FAIL = "[FAIL]"
_WARN = "[WARN]"


class _Checker:
    """Accumulates pass/fail/warn results and prints them as they are recorded."""

    def __init__(self) -> None:
        self.fail_count: int = 0
        self.warn_count: int = 0

    def check(
        self,
        label: str,
        ok: bool,
        detail: str,
        *,
        warn_only: bool = False,
    ) -> None:
        """Record and print a single check result.

        Args:
            label:     Short description of what is being checked.
            ok:        ``True`` if the check passed.
            detail:    Human-readable status string shown next to the label.
            warn_only: Treat failures as warnings instead of hard errors.
        """
        if ok:
            print(f"  {_PASS} {label}: {detail}")
        elif warn_only:
            self.warn_count += 1
            print(f"  {_WARN} {label}: {detail}")
        else:
            self.fail_count += 1
            print(f"  {_FAIL} {label}: {detail}")


def _try_import(module: str) -> tuple[bool, str]:
    """Attempt to import *module* and return ``(success, version_string)``."""
    try:
        mod = importlib.import_module(module)
        return True, str(getattr(mod, "__version__", "installed"))
    except ImportError:
        return False, "not installed"


def check_python(checker: _Checker) -> None:
    """Verify that the running Python version is within the supported range (3.9–3.13)."""
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    ok = v.major == 3 and 9 <= v.minor < 14
    checker.check("Python version", ok, ver_str if ok else f"{ver_str} — need 3.9 to 3.13")

    if sys.platform == "win32" and "LocalCache" in sys.executable:
        checker.check(
            "Python source",
            False,
            "Microsoft Store Python detected — can break 3LC project discovery. "
            "Install from python.org instead.",
            warn_only=True,
        )


def check_packages(checker: _Checker) -> None:
    """Verify that all required Python packages are importable."""
    required: list[tuple[str, str]] = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("3lc (tlc)", "tlc"),
        ("3lc-ultralytics", "tlc_ultralytics"),
        ("ultralytics", "ultralytics"),
        ("umap-learn", "umap"),
        ("PyYAML", "yaml"),
        ("tqdm", "tqdm"),
    ]
    for display, module in required:
        ok, detail = _try_import(module)
        if not ok and module == "umap":
            detail += " — pip install umap-learn (training will crash without it)"
        checker.check(display, ok, detail)


def check_gpu(checker: _Checker) -> None:
    """Check CUDA availability and report the GPU name."""
    try:
        import torch

        if torch.cuda.is_available():
            checker.check("CUDA GPU", True, torch.cuda.get_device_name(0))
        else:
            checker.check(
                "CUDA GPU",
                False,
                "not available — training will run on CPU (much slower)",
                warn_only=True,
            )
    except Exception as exc:
        checker.check("CUDA GPU", False, str(exc), warn_only=True)


def check_pytorch_cuda_build(checker: _Checker) -> None:
    """Detect CPU-only PyTorch installed on a machine that has a CUDA-capable GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            cuda_version = getattr(torch.version, "cuda", None)
            if cuda_version is None:
                checker.check(
                    "PyTorch CUDA build",
                    False,
                    f"torch {torch.__version__} is CPU-only. "
                    "If you have a GPU: pip uninstall torch torchvision, "
                    "then reinstall with --index-url for your CUDA version.",
                    warn_only=True,
                )
            else:
                checker.check(
                    "PyTorch CUDA build", True, f"CUDA {cuda_version} (driver may be missing)"
                )
        else:
            checker.check(
                "PyTorch CUDA build", True, f"CUDA {getattr(torch.version, 'cuda', 'unknown')}"
            )
    except ImportError:
        checker.check("PyTorch CUDA build", False, "torch not installed", warn_only=True)


def check_data(checker: _Checker) -> None:
    """Verify that all required starter-kit files and data directories are present."""
    required_files = [
        "config.yaml",
        "dataset.yaml",
        "sample_submission.csv",
        "register_tables.py",
        "train.py",
        "predict.py",
    ]
    required_dirs = [
        "data/train/images",
        "data/train/labels",
        "data/val/images",
        "data/val/labels",
        "data/test/images",
    ]

    for fname in required_files:
        p = KIT_DIR / fname
        checker.check(f"File: {fname}", p.is_file(), "found" if p.is_file() else "MISSING")

    for dname in required_dirs:
        p = KIT_DIR / dname
        if p.is_dir():
            count = sum(1 for _ in p.iterdir())
            checker.check(f"Dir:  {dname}", True, f"{count} files")
        else:
            checker.check(f"Dir:  {dname}", False, "MISSING")


def main() -> int:
    """Run all environment checks and return a non-zero exit code on failure."""
    print("=" * 65)
    print("  ENVIRONMENT CHECK — run before register_tables.py")
    print("=" * 65)

    checker = _Checker()

    print("\n-- Python --")
    check_python(checker)

    print("\n-- Packages --")
    check_packages(checker)

    print("\n-- GPU --")
    check_gpu(checker)
    check_pytorch_cuda_build(checker)

    print("\n-- Starter kit files --")
    check_data(checker)

    print("\n" + "=" * 65)
    if checker.fail_count == 0 and checker.warn_count == 0:
        print("  ALL CHECKS PASSED — ready to run register_tables.py")
    elif checker.fail_count == 0:
        print(f"  PASSED with {checker.warn_count} warning(s) — review above, then proceed")
    else:
        print(
            f"  {checker.fail_count} FAILED, {checker.warn_count} warning(s)"
            " — fix errors before proceeding"
        )
    print("=" * 65)
    return 1 if checker.fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
