#!/usr/bin/env python3
"""
Headless browser debug for pong2p ONNX page.
Run: pip install playwright && playwright install chromium && python debug_browser.py

Loads page, clicks Start, waits for frames, captures console logs and screenshots.
Output: debug-output/

Options: DEBUG_URL=... DEBUG_HEADED=1 DEBUG_WEBGPU=1 python debug_browser.py
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent / "debug-output"
WAIT_FRAMES = 15
TIMEOUT_MS = 120000
BASE = "https://wendlerc.github.io/pong2p"
HEADED = os.environ.get("DEBUG_HEADED", "").lower() in ("1", "true", "yes")
WEBGPU = os.environ.get("DEBUG_WEBGPU", "").lower() in ("1", "true", "yes")


async def main():
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Install: pip install playwright && playwright install chromium")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    logs = []

    def log(msg):
        line = f"[{__import__('datetime').datetime.now().isoformat()}] {msg}"
        logs.append(line)
        print(line)

    log(f"Launching Chromium (headed={HEADED})...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not HEADED)
        context = await browser.new_context()
        page = await context.new_page()

        def on_console(msg):
            log(f"CONSOLE {msg.type}: {msg.text}")

        page.on("console", on_console)

        url = os.environ.get("DEBUG_URL", f"{BASE}/onnx.html" + ("?webgpu=1" if WEBGPU else ""))
        log(f"Navigating to {url}")
        await page.goto(url, wait_until="networkidle", timeout=60000)

        log("Waiting for Start button...")
        await page.wait_for_selector("#startBtn:not([disabled])", timeout=90000)

        log("Clicking Start...")
        await page.click("#startBtn")

        log(f"Waiting ~{WAIT_FRAMES} frames (checking FPS every 2s)...")
        fps_history = []
        start = time.monotonic()
        for i in range(WAIT_FRAMES * 2):
            await asyncio.sleep(2)
            fps_el = await page.query_selector("#fpsValue")
            fps_text = await fps_el.text_content() if fps_el else "?"
            elapsed = (time.monotonic() - start) * 1000
            mem = await page.evaluate("""() => {
                if (performance.memory) return {used: performance.memory.usedJSHeapSize, total: performance.memory.totalJSHeapSize, limit: performance.memory.jsHeapSizeLimit};
                return null;
            }""")
            mem_mb = f", mem={mem['used']/1e6:.1f}MB" if mem else ""
            fps_history.append({"t_ms": int(elapsed), "fps": fps_text, "mem_mb": mem["used"]/1e6 if mem else None})
            log(f"  {elapsed/1000:.1f}s: FPS={fps_text}{mem_mb}")
            if fps_text == "err":
                log("FPS shows err - stopping")
                break
            if elapsed > TIMEOUT_MS:
                break

        log("Taking screenshot...")
        await page.screenshot(path=OUT_DIR / "final.png")

        log("Capturing canvas...")
        canvas_data = await page.evaluate("""() => {
            const c = document.getElementById('frame');
            return c ? c.toDataURL('image/png') : null;
        }""")
        if canvas_data:
            import base64
            b64 = canvas_data.split(",", 1)[-1]
            (OUT_DIR / "canvas.png").write_bytes(base64.b64decode(b64))

        await browser.close()

    (OUT_DIR / "log.txt").write_text("\n".join(logs))
    (OUT_DIR / "fps_history.json").write_text(json.dumps(fps_history, indent=2))
    log(f"Done. Output in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
