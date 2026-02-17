# Headless browser debug for pong2p ONNX

Debug the rendering/freeze issue without manual interaction.

**Python (recommended):**
```bash
# From toy-wm-private (has playwright):
cd ../toy-wm-private
pip install playwright && playwright install chromium
python ../wendlerc.github.io/pong2p/debug_browser.py
```

**Or Node:**
```bash
cd pong2p
npm install
npx playwright install chromium
npm run debug
```

**Output** in `pong2p/debug-output/`:
- `log.txt` - Console logs, FPS over time
- `final.png` - Full page screenshot
- `canvas.png` - Canvas content
- `fps_history.json` - FPS at each 2s interval

Headless run showed stable 0.7 FPS (no freeze) - freeze may be GPU/render specific.
