#!/usr/bin/env python3

import sys
from pathlib import Path

FILE = "time_integrator.cpp"
TARGET_LINES = [1837, 1839]   # 1-based line numbers


def toggle_line(line):
    stripped = line.lstrip()
    indent = line[:len(line) - len(stripped)]

    # If already commented → uncomment
    if stripped.startswith("//"):
        new = stripped[2:]
        if new.startswith(" "):
            new = new[1:]
        return indent + new
    else:
        return indent + "// " + stripped


def main():
    path = Path(FILE)
    lines = path.read_text().splitlines(keepends=True)

    print("---- BEFORE TOGGLE ----")
    for ln in TARGET_LINES:
        idx = ln - 1
        print(f"Line {ln}: {lines[idx].rstrip()}")

    for ln in TARGET_LINES:
        idx = ln - 1
        lines[idx] = toggle_line(lines[idx])

    print("\n---- AFTER TOGGLE ----")
    for ln in TARGET_LINES:
        idx = ln - 1
        print(f"Line {ln}: {lines[idx].rstrip()}")

    path.write_text("".join(lines))
    print(f"\nToggled lines {TARGET_LINES} in {FILE}")


if __name__ == "__main__":
    main()

