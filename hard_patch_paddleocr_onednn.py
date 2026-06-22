from pathlib import Path

root = Path(".venv/Lib/site-packages/paddleocr")
patched = []

for path in root.rglob("*.py"):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue

    original = text
    text = text.replace("config.enable_mkldnn();", "# HARD DISABLED: config.enable_mkldnn();")
    text = text.replace("config.enable_mkldnn()", "# HARD DISABLED: config.enable_mkldnn()")
    text = text.replace("config.enable_mkldnn_bfloat16()", "# HARD DISABLED: config.enable_mkldnn_bfloat16()")
    text = text.replace("config.set_mkldnn_cache_capacity(10)", "# HARD DISABLED: config.set_mkldnn_cache_capacity(10)")
    text = text.replace("config.set_mkldnn_cache_capacity(0)", "# HARD DISABLED: config.set_mkldnn_cache_capacity(0)")

    if text != original:
        backup = path.with_suffix(path.suffix + ".bak_onednn")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        patched.append(str(path))

print("Patched files:")
for p in patched:
    print(p)

if not patched:
    print("No PaddleOCR mkldnn calls found to patch.")
