"""Download benchmark files into ``data/raw`` (idempotent and atomic)."""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

_FINQA = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"
_FINANCEBENCH = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data"

DATASETS: dict[str, tuple[str, str]] = {
    "finqa-train": (f"{_FINQA}/train.json", "finqa_train.json"),
    "finqa-dev": (f"{_FINQA}/dev.json", "finqa_dev.json"),
    "finqa-test": (f"{_FINQA}/test.json", "finqa_test.json"),
    "financebench": (f"{_FINANCEBENCH}/financebench_open_source.jsonl", "financebench.jsonl"),
}


def download(name: str, data_dir: Path, *, force: bool = False) -> Path:
    """Fetch dataset ``name`` into ``data_dir`` and return its path."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Choose from: {', '.join(DATASETS)}")
    url, filename = DATASETS[name]
    dest = data_dir / filename
    if dest.exists() and not force:
        logger.info("%s already present at %s", name, dest)
        return dest

    data_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=data_dir, suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("Downloaded %s -> %s", name, dest)
    return dest


_FINANCEBENCH_PDFS = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"


def download_financebench_pdfs(doc_names: Iterable[str], pdf_dir: Path) -> list[Path]:
    """Download the named FinanceBench filings (skipping ones already present)."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in sorted(set(doc_names)):
        dest = pdf_dir / f"{name}.pdf"
        if not dest.exists():
            fd, tmp = tempfile.mkstemp(dir=pdf_dir, suffix=".part")
            os.close(fd)
            try:
                urllib.request.urlretrieve(f"{_FINANCEBENCH_PDFS}/{name}.pdf", tmp)
                os.replace(tmp, dest)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            logger.info("Downloaded %s", dest.name)
        paths.append(dest)
    return paths
