#!/usr/bin/env python3
"""Builds docs/index.html from web/template.html with the trained model, the test-set
precision/recall curve and the test results embedded, so the page works on its own."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
page = (root / "web" / "template.html").read_text(encoding="utf-8")
for marker, path in [("MODEL", "docs/model.json"), ("CURVE", "docs/curve.json"), ("RESULTS", "results/test.json")]:
    data = json.dumps(json.loads((root / path).read_text(encoding="utf-8")), separators=(",", ":"))
    page = page.replace(f"/*{marker}*/null", data.replace("</", "<\\/"))
assert "/*" not in page.split("<script>")[1].split("const ADDONS")[0], "a placeholder was not filled"
(root / "docs" / "index.html").write_text(page, encoding="utf-8")
print(f"Wrote docs/index.html ({len(page):,} bytes)")
