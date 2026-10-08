"""Check Git publication candidates and a release ZIP without exposing secret values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|"
    r"\b(?:ghp_|github_pat_|hf_|sk-proj-|sk-)[A-Za-z0-9]{20,}|\bAKIA[A-Z0-9]{16}\b"
)
ASSIGNMENT = re.compile(
    r"(?i)\b(?:[A-Z_]*(?:API_KEY|ACCESS_TOKEN|SECRET_KEY)|password)\s*=\s*"
    r"(?:[\"']([^\"'\n]{8,})[\"']|([^\s,;\n]{12,}))"
)
BLOCKED_PARTS = {".git", ".release-work", "__pycache__", ".pytest_cache", "checkpoints", "swanlog", "wandb", "logs", "outputs"}
BLOCKED_SUFFIXES = {".pem", ".key", ".pyc", ".parquet", ".whl", ".pt", ".pth", ".safetensors", ".log"}


def inspect(name: str, data: bytes, forbidden: list[str]) -> list[str]:
    path = Path(name)
    errors = []
    if BLOCKED_PARTS.intersection(path.parts) or path.suffix in BLOCKED_SUFFIXES or path.name in {".env", "id_rsa", "id_ed25519"} or (path.name.startswith(".env.") and path.name != ".env.example"):
        errors.append(f"Excluded artifact: {name}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return errors
    if any(value.lower() in text.lower() for value in forbidden):
        errors.append(f"Forbidden identifier: {name}")
    if SECRET.search(text):
        errors.append(f"Possible credential: {name}")
    for match in ASSIGNMENT.finditer(text):
        value = match.group(1) or match.group(2)
        if value.startswith(("$", "{", "os.", "self.", "config.", "None", "spotify_", "login_")) or re.search(r"(?i)YOUR_|example|placeholder|token-abc123|API_KEY|ACCESS_TOKEN|SECRET_KEY", value):
            continue
        # Only literal assignments warrant review; function calls and variables
        # are not credentials stored in the source.
        if match.group(1) and re.fullmatch(r"[A-Za-z0-9_+/=.-]{16,}", value):
            errors.append(f"Review literal credential assignment: {name}")
    if path.suffix == ".ipynb":
        notebook = json.loads(text)
        if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in notebook.get("cells", [])):
            errors.append(f"Saved notebook execution state: {name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbid", action="append", default=[], help="Private identifier to reject (never printed).")
    parser.add_argument("--archive", type=Path, help="Also scan a release ZIP.")
    args = parser.parse_args()
    git_root = Path(subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"], text=True).strip())
    names = subprocess.check_output(["git", "-C", str(git_root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]).decode().split("\0")
    errors = []
    count = 0
    for name in sorted(set(names) - {""}):
        path = git_root / name
        if path.is_file():
            errors.extend(inspect(name, path.read_bytes(), args.forbid))
            count += 1
    # Scan staged blobs as well: the index can differ from the working tree.
    staged = subprocess.check_output(["git", "-C", str(git_root), "ls-files", "-z"]).decode().split("\0")
    for name in set(staged) - {""}:
        data = subprocess.check_output(["git", "-C", str(git_root), "show", f":{name}"])
        errors.extend(inspect(name, data, args.forbid))
    if args.archive:
        with zipfile.ZipFile(args.archive) as archive:
            if archive.testzip():
                errors.append("Corrupted ZIP entry")
            for name in archive.namelist():
                if not name.endswith("/"):
                    errors.extend(inspect(name, archive.read(name), args.forbid))
    if errors:
        raise SystemExit("\n".join(sorted(set(errors))))
    print(f"PASS: {count} Git publication candidates and staged content" + ("; release ZIP" if args.archive else ""))


if __name__ == "__main__":
    main()
