test:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 - <<'PY'
    import json
    from pathlib import Path

    json_notebooks = [
        Path("notebooks/curves/HandshakeCurves.ipynb"),
        Path("notebooks/curves/curve-ideal-scratchpad.ipynb"),
        Path("notebooks/empty-or-broken/empty-python-notebook.ipynb"),
        Path("notebooks/lattices/integral-lattice-import-failure.ipynb"),
    ]
    for path in json_notebooks:
        data = json.loads(path.read_text())
        if "cells" not in data:
            raise SystemExit(f"{path} is missing notebook cells")

    html_export = Path("notebooks/periods/fermat-periods-nbviewer.html")
    if not html_export.read_text(errors="replace").lstrip().startswith("<!DOCTYPE html>"):
        raise SystemExit(f"{html_export} is not the expected HTML notebook export")

    lock_files = list(Path("references/generated-indexes").rglob("*.lock"))
    if lock_files:
        joined = "\n".join(str(path) for path in lock_files)
        raise SystemExit(f"generated index lock files should not be tracked:\n{joined}")

    local_link = Path("references/local-system-dependencies/mylibs/libflint.so.22")
    if local_link.exists() or local_link.is_symlink():
        raise SystemExit(f"machine-local FLINT symlink was imported: {local_link}")
    PY

