# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chris Wendler's academic personal website hosted on GitHub Pages at `wendlerc.github.io`. It combines a space-themed interactive homepage with educational notes, ARENA-style exercises, and a neural Pong demo.

There is **no build system** — this is a static site served directly by GitHub Pages (no Jekyll, no bundler). HTML files in `notes/` are generated from Markdown (likely via Pandoc) but the conversion is done manually, not by CI.

## Key Areas

- **`index.html` + `styles.css` + `js/`**: The homepage. An interactive star-field animation with game mechanics (phases, power-ups, collision detection). `js/main.js` is the core game loop; `Star.js`, `Comet.js`, `Nebula.js`, `StarColors.js` are supporting modules.
- **`notes/`**: Educational content — short notes, long-form tutorials, exercises (`.html` + `.md` pairs). Exercises (`ex1-ex4`) cover KV-cache, flow matching, FAR video models. The index is `notes/index.md`.
- **`solutions/`**: Python solution files for the exercises.
- **`pong2p/`**: Neural Pong demo using ONNX Runtime + WebGPU/WASM. Contains large `.onnx` model files tracked via Git LFS (see `.gitattributes`). Has its own `package.json` for Playwright-based debugging (`debug-browser.mjs`).
- **`resources/`**: PDFs, bibliography (`llm.bib`), images for notes subdirectories.

## Development

No build commands needed — open HTML files directly in a browser. For the Pong demo:

```bash
cd pong2p && npm install   # one-time
npx playwright test        # or use debug scripts in package.json
```

Notes workflow: edit `.md` files, then generate `.html` (e.g., via Pandoc). Both files are committed.

## Git Conventions

- Large model files (`.onnx`) use Git LFS — see `.gitattributes` for configuration. Exception: `pong2p_dmd.onnx` is explicitly excluded from LFS.
- The `paper/` directory is a symlink (gitignored).
- Don't commit large binary files — check `.gitignore`.

## External Dependencies (CDN)

The homepage loads Google Fonts (Source Sans Pro, Orbitron), FontAwesome 6.1.1, and Academicons 1.9.2 from CDNs. No local package management for the main site.
