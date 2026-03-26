"""
Generate Jupyter notebooks from the HTML exercises and Python solution files.

Produces:
  notebooks/ex1_kv_cache.ipynb              (exercise stubs)
  notebooks/ex2_flow_matching_mnist.ipynb
  notebooks/ex3_far_pong.ipynb
  notebooks/ex4_far_kv_cache.ipynb
  notebooks/ex1_kv_cache_solutions.ipynb    (complete solutions)
  notebooks/ex2_flow_matching_mnist_solutions.ipynb
  notebooks/ex3_far_pong_solutions.ipynb
  notebooks/ex4_far_kv_cache_solutions.ipynb
"""

import json, os, re
from html.parser import HTMLParser
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_cell(cell_type, source, **kwargs):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else source.split("\n"),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def make_notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "cells": cells,
    }


def source_lines(text):
    """Convert a string into a list of newline-terminated lines (ipynb format)."""
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


# ── HTML Parser ──────────────────────────────────────────────────────────────

class ExerciseHTMLParser(HTMLParser):
    """Extract structured content from the exercise HTML files."""

    def __init__(self):
        super().__init__()
        self.cells = []
        self._stack = []       # tag stack
        self._text = ""
        self._in_pre = False
        self._in_code = False
        self._in_details = False
        self._details_summary = ""
        self._details_body = ""
        self._current_exercise = None
        self._collecting_md = ""  # markdown accumulator
        self._in_style = False
        self._in_script = False
        self._skip_tags = {"style", "script"}
        self._in_skip = False

    def _flush_md(self):
        txt = self._collecting_md.strip()
        if txt:
            self.cells.append(("md", txt))
        self._collecting_md = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag in self._skip_tags:
            self._in_skip = True
            return

        self._stack.append(tag)

        if tag == "pre":
            self._in_pre = True
            self._text = ""
        elif tag == "code" and self._in_pre:
            self._in_code = True
            self._text = ""
        elif tag == "details":
            self._in_details = True
            self._details_summary = ""
            self._details_body = ""
        elif tag == "summary" and self._in_details:
            self._text = ""

        # Convert HTML tags to markdown
        if not self._in_pre and not self._in_details:
            if tag == "h1":
                self._flush_md()
                self._collecting_md += "# "
            elif tag == "h2":
                self._flush_md()
                self._collecting_md += "\n## "
            elif tag == "h3":
                self._flush_md()
                self._collecting_md += "\n### "
            elif tag == "h4":
                self._flush_md()
                self._collecting_md += "\n#### "
            elif tag == "p":
                self._collecting_md += "\n\n"
            elif tag == "li":
                self._collecting_md += "\n- "
            elif tag == "ol":
                pass  # handled by li
            elif tag == "ul":
                pass
            elif tag == "strong" or tag == "b":
                self._collecting_md += "**"
            elif tag == "em" or tag == "i":
                self._collecting_md += "*"
            elif tag == "code" and not self._in_pre:
                self._collecting_md += "`"
            elif tag == "a":
                href = attrs_dict.get("href", "")
                self._collecting_md += "["
            elif tag == "br":
                self._collecting_md += "\n"
            elif tag == "div":
                cls = attrs_dict.get("class", "")
                if "exercise" in cls:
                    self._flush_md()
                    self._collecting_md += "\n---\n\n"
                elif "learning-obj" in cls:
                    self._flush_md()
                    self._collecting_md += "\n> **Learning objectives:**\n"
                elif "note" in cls:
                    self._flush_md()
                    self._collecting_md += "\n> **Note:** "
            elif tag == "blockquote":
                self._collecting_md += "\n> "
            elif tag == "table":
                self._flush_md()
                self._collecting_md += "\n"

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._in_skip = False
            return

        if self._stack and self._stack[-1] == tag:
            self._stack.pop()

        if tag == "code" and self._in_pre and self._in_code:
            self._in_code = False
        elif tag == "pre" and self._in_pre:
            self._in_pre = False
            code = self._text.strip()
            if self._in_details:
                self._details_body = code
            else:
                # Flush any pending markdown, then add code cell
                self._flush_md()
                self.cells.append(("code", code))
            self._text = ""
        elif tag == "summary" and self._in_details:
            self._details_summary = self._text.strip()
            self._text = ""
        elif tag == "details":
            self._in_details = False
            if self._details_summary.lower() in ("solution", "solutions"):
                self.cells.append(("solution", self._details_body))
            elif self._details_summary.lower().startswith("hint"):
                self._collecting_md += f"\n\n<details>\n<summary>{self._details_summary}</summary>\n\n{self._details_body}\n\n</details>\n"
            self._details_summary = ""
            self._details_body = ""

        if not self._in_pre and not self._in_details:
            if tag in ("strong", "b"):
                self._collecting_md += "**"
            elif tag in ("em", "i"):
                self._collecting_md += "*"
            elif tag == "code":
                self._collecting_md += "`"
            elif tag == "a":
                # try to grab href - simplified
                self._collecting_md += "]()"
            elif tag in ("h1", "h2", "h3", "h4"):
                self._collecting_md += "\n"
            elif tag == "p":
                pass

    def handle_data(self, data):
        if self._in_skip:
            return
        if self._in_pre or (self._in_details and self._in_code):
            self._text += data
        elif self._in_details:
            if not self._in_code:
                self._details_body += data
        else:
            self._collecting_md += data

    def handle_entityref(self, name):
        entities = {"larr": "\u2190", "rarr": "\u2192", "mdash": "\u2014",
                     "ndash": "\u2013", "times": "\u00d7", "le": "\u2264",
                     "ge": "\u2265", "amp": "&", "lt": "<", "gt": ">",
                     "nbsp": " ", "lfloor": "\u230a", "rfloor": "\u230b"}
        ch = entities.get(name, f"&{name};")
        if self._in_pre:
            self._text += ch
        elif self._in_details:
            self._details_body += ch
        else:
            self._collecting_md += ch

    def handle_charref(self, name):
        ch = chr(int(name, 16) if name.startswith("x") else int(name))
        if self._in_pre:
            self._text += ch
        elif self._in_details:
            self._details_body += ch
        else:
            self._collecting_md += ch

    def finish(self):
        self._flush_md()
        return self.cells


def parse_html(filepath):
    """Parse an exercise HTML file and return list of (type, content) tuples."""
    parser = ExerciseHTMLParser()
    with open(filepath, "r") as f:
        parser.feed(f.read())
    return parser.finish()


def clean_md(text):
    """Clean up extracted markdown."""
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove leading/trailing whitespace per line, preserving structure
    lines = text.split("\n")
    # Don't strip lines that are part of list items or blockquotes
    cleaned = []
    for line in lines:
        cleaned.append(line.rstrip())
    text = "\n".join(cleaned)
    return text.strip()


# ── Notebook generators ──────────────────────────────────────────────────────

def make_exercise_notebook(html_path):
    """Create an exercise notebook (stubs only, no solutions) from HTML."""
    parsed = parse_html(html_path)
    cells = []

    for cell_type, content in parsed:
        if cell_type == "md":
            md = clean_md(content)
            if md:
                cells.append(make_cell("markdown", source_lines(md)))
        elif cell_type == "code":
            cells.append(make_cell("code", source_lines(content)))
        elif cell_type == "solution":
            # In exercise notebook: add collapsed solution as markdown
            cells.append(make_cell("markdown", source_lines(
                "<details>\n<summary><b>Solution</b></summary>\n\n```python\n"
                + content + "\n```\n\n</details>"
            )))

    return make_notebook(cells)


def make_solution_notebook_from_py(py_path):
    """Create a solution notebook from a Python solution file.

    Splits the file into cells at comment markers like:
      # ── Exercise N: Title ──
    and at test functions / __main__ blocks.
    """
    with open(py_path, "r") as f:
        source = f.read()

    cells = []
    # Split on section markers
    sections = re.split(r'(# ── .+? ──+)', source)

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        if section.startswith("# ── "):
            # Section header -> markdown
            title = section.strip("# ─ ").strip()
            cells.append(make_cell("markdown", source_lines(f"## {title}")))
        elif section.startswith('"""'):
            # Module docstring + imports section
            # Split into docstring (markdown) and code (imports)
            doc_end = section.find('"""', 3)
            if doc_end >= 0:
                doc = section[3:doc_end].strip()
                # Title from first line
                title_line = doc.split("\n")[0].rstrip(" =\n")
                cells.append(make_cell("markdown", source_lines(f"# {title_line}")))
                # Remaining code after docstring (imports etc.)
                code_after = section[doc_end + 3:].strip()
                if code_after:
                    cells.append(make_cell("code", source_lines(code_after)))
            else:
                cells.append(make_cell("code", source_lines(section)))
        elif "def run_all_tests():" in section:
            # Test section: split into the function def and a call cell
            cells.append(make_cell("code", source_lines(section)))
            cells.append(make_cell("code", source_lines("run_all_tests()")))
        elif 'if __name__ == "__main__":' in section:
            # Skip the __main__ block (we add run_all_tests() above)
            pass
        else:
            cells.append(make_cell("code", source_lines(section)))

    return make_notebook(cells)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    base = Path(__file__).parent
    out_dir = base / "notebooks"
    out_dir.mkdir(exist_ok=True)

    exercises = [
        ("ex1-kv-cache", "ex1_kv_cache", "ex1_kv_cache"),
        ("ex2-flow-matching-mnist", "ex2_flow_matching_mnist", "ex2_flow_matching"),
        ("ex3-far-pong", "ex3_far_pong", "ex3_far_pong"),
        ("ex4-far-kv-cache", "ex4_far_kv_cache", "ex4_far_kv_cache"),
    ]

    for html_name, nb_name, py_name in exercises:
        html_path = base / "notes" / f"{html_name}.html"
        py_path = base / "solutions" / f"{py_name}_solutions.py"

        # Exercise notebook
        print(f"Generating notebooks/{nb_name}.ipynb ...")
        nb = make_exercise_notebook(str(html_path))
        with open(out_dir / f"{nb_name}.ipynb", "w") as f:
            json.dump(nb, f, indent=1)

        # Solution notebook
        print(f"Generating notebooks/{nb_name}_solutions.ipynb ...")
        nb = make_solution_notebook_from_py(str(py_path))
        with open(out_dir / f"{nb_name}_solutions.ipynb", "w") as f:
            json.dump(nb, f, indent=1)

    print(f"\nDone! Generated {len(exercises) * 2} notebooks in {out_dir}/")


if __name__ == "__main__":
    main()
