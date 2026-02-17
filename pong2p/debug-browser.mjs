#!/usr/bin/env node
/**
 * Headless browser debug for pong2p ONNX page.
 * Run: npx playwright install chromium && node debug-browser.mjs
 * Or: npm install playwright && node debug-browser.mjs
 *
 * Loads page, clicks Start, waits for frames, captures console logs and screenshots.
 * Output: debug-output/
 */
import { chromium } from 'playwright';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';

const BASE = 'https://wendlerc.github.io/pong2p';
const OUT_DIR = join(process.cwd(), 'debug-output');
const WAIT_FRAMES = 15;
const TIMEOUT_MS = 120000;

async function main() {
  mkdirSync(OUT_DIR, { recursive: true });
  const logs = [];
  const log = (msg) => {
    const line = `[${new Date().toISOString()}] ${msg}`;
    logs.push(line);
    console.log(line);
  };

  log('Launching Chromium...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  page.on('console', (msg) => {
    const text = msg.text();
    const type = msg.type();
    log(`CONSOLE ${type}: ${text}`);
  });
  page.on('pageerror', (err) => log(`PAGE ERROR: ${err.message}`));

  const url = process.argv[2] || `${BASE}/onnx.html`;
  log(`Navigating to ${url}`);
  await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });

  log('Waiting for Start button to be enabled...');
  await page.waitForSelector('#startBtn:not([disabled])', { timeout: 90000 });

  log('Clicking Start...');
  await page.click('#startBtn');

  log(`Waiting for ~${WAIT_FRAMES} frames (checking FPS every 2s)...`);
  const fpsHistory = [];
  const startTime = Date.now();
  for (let i = 0; i < WAIT_FRAMES * 2; i++) {
    await page.waitForTimeout(2000);
    const fpsEl = await page.$('#fpsValue');
    const fpsText = fpsEl ? await fpsEl.textContent() : '?';
    fpsHistory.push({ t: Date.now() - startTime, fps: fpsText });
    log(`  ${(Date.now() - startTime) / 1000}s: FPS=${fpsText}`);
    if (fpsText === 'err') {
      log('FPS shows err - stopping');
      break;
    }
    if (Date.now() - startTime > TIMEOUT_MS) break;
  }

  log('Taking screenshot...');
  await page.screenshot({ path: join(OUT_DIR, 'final.png') });

  log('Capturing canvas as image...');
  const canvasData = await page.evaluate(() => {
    const c = document.getElementById('frame');
    if (!c) return null;
    return c.toDataURL('image/png');
  });
  if (canvasData) {
    const base64 = canvasData.replace(/^data:image\/png;base64,/, '');
    writeFileSync(join(OUT_DIR, 'canvas.png'), Buffer.from(base64, 'base64'));
  }

  await browser.close();

  writeFileSync(join(OUT_DIR, 'log.txt'), logs.join('\n'));
  writeFileSync(join(OUT_DIR, 'fps_history.json'), JSON.stringify(fpsHistory, null, 2));
  log(`Done. Output in ${OUT_DIR}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
