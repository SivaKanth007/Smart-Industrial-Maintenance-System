"""
Dashboard Screenshot Capture
==============================
Starts Streamlit on a background port, navigates to each page, and saves
PNG screenshots to assets/. Called automatically at the end of run_pipeline.py.

Requirements:
    pip install playwright
    playwright install chromium
"""

import os
import sys
import time
import socket
import subprocess

# Force UTF-8 for all subprocess communication on Windows (Streamlit uses → in output)
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Reconfigure current process stdio to UTF-8 so print() handles → safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
APP_PATH = os.path.join(PROJECT_ROOT, "dashboard", "app.py")
PORT = 8502  # Separate port — avoids conflicts with a live dashboard on 8501

# Pages to capture: (sidebar label, output filename)
PAGES = [
    ("Fleet Overview",              "dashboard_fleet_overview.png"),
    ("Risk Assessment",             "dashboard_risk_assessment.png"),
    ("Maintenance Schedule",        "dashboard_maintenance_schedule.png"),
    ("Model Performance",           "dashboard_model_performance.png"),
    ("Explainability & AI Insights","dashboard_explainability.png"),
    ("Maintenance History",         "dashboard_maintenance_history.png"),
    ("Operational Context",         "dashboard_operational_context.png"),
]


def _wait_for_port(port: int, timeout: int = 45) -> bool:
    """Poll until Streamlit is accepting connections or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def capture(verbose: bool = True) -> bool:
    """
    Capture screenshots of every dashboard page and write them to assets/.

    Returns True on success, False if playwright is not installed or the
    Streamlit process fails to start.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        print(
            "[SCREENSHOTS] playwright not installed — skipping auto-capture.\n"
            "[SCREENSHOTS] To enable: pip install playwright && playwright install chromium"
        )
        return False

    os.makedirs(ASSETS_DIR, exist_ok=True)

    # ── Launch Streamlit ──────────────────────────────────────────────────────
    cmd = [
        sys.executable, "-m", "streamlit", "run", APP_PATH,
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--server.runOnSave", "false",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )

    try:
        if verbose:
            print(f"[SCREENSHOTS] Starting Streamlit on port {PORT} ...")
        if not _wait_for_port(PORT, timeout=45):
            print("[SCREENSHOTS] Streamlit did not start in time — skipping screenshots.")
            return False

        time.sleep(3)  # Let the initial render settle

        # ── Playwright session ────────────────────────────────────────────────
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 900})

            page.goto(f"http://localhost:{PORT}", timeout=30_000)
            page.wait_for_load_state("networkidle", timeout=30_000)
            time.sleep(2)

            for label, filename in PAGES:
                if verbose:
                    print(f"[SCREENSHOTS] Capturing: {label} ...")
                try:
                    # Click the matching sidebar radio label
                    sidebar = page.locator("[data-testid='stSidebar']")
                    sidebar.get_by_text(label, exact=True).click()
                    page.wait_for_load_state("networkidle", timeout=15_000)
                    time.sleep(2)  # Wait for Plotly charts to finish rendering

                    out_path = os.path.join(ASSETS_DIR, filename)
                    page.screenshot(path=out_path, full_page=False)
                    if verbose:
                        print(f"[SCREENSHOTS]   Saved → {filename}")
                except PWTimeout:
                    print(f"[SCREENSHOTS]   Timeout on '{label}' — skipping.")
                except Exception as exc:
                    safe_msg = str(exc).encode("ascii", errors="replace").decode("ascii")
                    print(f"[SCREENSHOTS]   Error on '{label}': {safe_msg}")

            browser.close()

        if verbose:
            print(f"[SCREENSHOTS] Done. Screenshots saved to {ASSETS_DIR}/")
        return True

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    success = capture()
    sys.exit(0 if success else 1)
