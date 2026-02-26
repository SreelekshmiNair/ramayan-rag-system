import json
import shutil
from pathlib import Path


def fix_notebook(path: Path):
    path = Path(path)
    if not path.exists():
        print(f"Notebook not found: {path}")
        return 1

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copyfile(path, backup)
    print(f"Backup saved to: {backup}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure top-level metadata exists and remove widgets if present
    metadata = data.get("metadata") or {}
    if isinstance(metadata, dict) and "widgets" in metadata:
        metadata.pop("widgets", None)
    data["metadata"] = metadata

    # Ensure nbformat fields exist
    data.setdefault("nbformat", 4)
    data.setdefault("nbformat_minor", 5)

    # Fix cells: move cell-level "id" into metadata.id and remove any metadata.widgets
    cells = data.get("cells", [])
    for cell in cells:
        # ensure metadata exists
        cell_meta = cell.get("metadata") or {}

        # if cell-level id exists, move it into metadata.id
        if "id" in cell:
            # avoid overwriting an existing metadata.id
            if "id" not in cell_meta:
                cell_meta["id"] = cell.pop("id")
            else:
                # remove the top-level id if redundant
                cell.pop("id", None)

        # remove widgets from cell metadata if present
        if isinstance(cell_meta, dict) and "widgets" in cell_meta:
            cell_meta.pop("widgets", None)

        cell["metadata"] = cell_meta

    data["cells"] = cells

    # Write cleaned notebook back
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Fixed notebook written to: {path}")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: fix_notebook_widgets.py <path-to-notebook>")
        sys.exit(2)
    notebook_path = Path(sys.argv[1])
    sys.exit(fix_notebook(notebook_path))
