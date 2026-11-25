"""Make the poster's QR codes.

    .venv\\Scripts\\python.exe scripts\\make_qr.py demo <url>
    .venv\\Scripts\\python.exe scripts\\make_qr.py github <url>
Writes final_ie643_report/figures/<name>_qr.png (needs the `segno` package).
"""

from __future__ import annotations

import sys
from pathlib import Path

import segno

OUT = Path(__file__).resolve().parents[1] / "final_ie643_report" / "figures"


def main() -> None:
    name, url = sys.argv[1], sys.argv[2]
    qr = segno.make(url, error="m", micro=False)          # medium error correction
    path = OUT / f"{name}_qr.png"
    qr.save(path, scale=20, border=4, dark="#000000", light="#FFFFFF")   # 4-module quiet zone
    print(path, "version", qr.version, "modules", qr.symbol_size(scale=1, border=0)[0])


if __name__ == "__main__":
    main()
