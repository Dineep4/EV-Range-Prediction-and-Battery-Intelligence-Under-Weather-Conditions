
from pathlib import Path
import json
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "NASA"
OUTPUT = ROOT / "data" / "raw_inventory.json"


def get_field(obj, name, default=None):
    """Read a field from a MATLAB struct."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def inspect_mat(path):
    result = {
        "file": str(path.relative_to(ROOT)),
        "status": "ok",
    }

    try:
        mat = sio.loadmat(
            str(path),
            squeeze_me=True,
            struct_as_record=False,
        )
        keys = [k for k in mat if not k.startswith("__")]
        result["variables"] = keys

        battery_key = next(
    (k for k in keys if k.lower() in ("battery", "data")),
    keys[0] if keys else None,
)

        if battery_key is None:
            result["note"] = "No obvious battery variable; inspect keys."
            return result

        battery = mat[battery_key]
        cycles = get_field(battery, "cycle")

        if cycles is None:
            result["note"] = (
                f"Variable '{battery_key}' has no cycle field."
            )
            return result

        if not hasattr(cycles, "__iter__") or isinstance(cycles, str):
            cycles = [cycles]

        result["cycle_count"] = len(cycles)
        types = {}
        capacity_count = 0
        examples = []

        for cycle in cycles:
            kind = get_field(cycle, "type", "unknown")
            kind = str(kind)
            types[kind] = types.get(kind, 0) + 1

            data = get_field(cycle, "data")
            capacity = get_field(data, "Capacity") if data is not None else None

            if capacity is not None:
                capacity_count += 1
                if len(examples) < 3:
                    try:
                        examples.append(float(capacity))
                    except (TypeError, ValueError):
                        examples.append(str(capacity))

        result["cycle_types"] = types
        result["cycles_with_capacity"] = capacity_count
        result["capacity_examples_Ah"] = examples

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


def main():
    if not DATA_DIR.exists():
        print(f"NASA folder not found: {DATA_DIR}")
        return

    files = sorted(DATA_DIR.rglob("*.mat"))

    if not files:
        print(f"No .mat files found under: {DATA_DIR}")
        print("Check that your files have been extracted.")
        return

    print(f"Found {len(files)} MATLAB files.\n")
    results = []

    for path in files:
        result = inspect_mat(path)
        results.append(result)
        print(f"File: {result['file']}")
        print(f"Status: {result['status']}")
        print(f"Variables: {result.get('variables', [])}")
        print(f"Cycles: {result.get('cycle_count', 'unknown')}")
        print(f"Cycle types: {result.get('cycle_types', {})}")
        print(
            "Cycles with capacity:",
            result.get("cycles_with_capacity", "unknown"),
        )
        if "error" in result:
            print("Error:", result["error"])
        print("-" * 55)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"\nInventory saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
