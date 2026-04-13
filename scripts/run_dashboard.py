#!/usr/bin/env python
"""
Dashboard launcher

Usage:
    python scripts/run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    dashboard_path = (
        Path(__file__).parent.parent / "src" / "dashboard" / "app.py"
    )

    print(f"Starting Streamlit dashboard: {dashboard_path}")
    print("Open http://localhost:8501 in your browser\n")

    subprocess.run(
        ["streamlit", "run", str(dashboard_path), "--server.port", "8501"],
    )


if __name__ == "__main__":
    main()
