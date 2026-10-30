"""Atomic JSON records and file fingerprints for resumable comparison artifacts."""

import hashlib
import json


def read_json(path):
    contents = path.read_text()
    values = json.loads(contents)
    return values


def write_json(path, values):
    """Publish a record only after its complete contents have been written."""
    contents = json.dumps(values, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents + "\n")
    temporary.replace(path)


def file_sha256(path):
    """Fingerprint final weights without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    result = digest.hexdigest()
    return result
