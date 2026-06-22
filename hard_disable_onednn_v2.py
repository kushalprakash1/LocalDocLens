from pathlib import Path
import py_compile

root = Path(".venv/Lib/site-packages/paddleocr")
patched = []

for path in root.rglob("*.py"):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue

    original = text

    # Fix the broken empty-if issue caused by replacing real code with comment-only lines.
    text = text.replace("# HARD DISABLED: config.enable_mkldnn();", "pass  # HARD DISABLED: config.enable_mkldnn();")
    text = text.replace("# HARD DISABLED: config.enable_mkldnn()", "pass  # HARD DISABLED: config.enable_mkldnn()")
    text = text.replace("# HARD DISABLED: config.enable_mkldnn_bfloat16()", "pass  # HARD DISABLED: config.enable_mkldnn_bfloat16()")
    text = text.replace("# HARD DISABLED: config.set_mkldnn_cache_capacity(10)", "pass  # HARD DISABLED: config.set_mkldnn_cache_capacity(10)")
    text = text.replace("# HARD DISABLED: config.set_mkldnn_cache_capacity(0)", "pass  # HARD DISABLED: config.set_mkldnn_cache_capacity(0)")

    # Hard-disable the conditional path itself.
    text = text.replace("if args.enable_mkldnn:", "if False:  # HARD DISABLED: args.enable_mkldnn")
    text = text.replace("if enable_mkldnn:", "if False:  # HARD DISABLED: enable_mkldnn")

    # If any defaults force mkldnn on, force them off.
    text = text.replace("enable_mkldnn=True", "enable_mkldnn=False")
    text = text.replace("enable_mkldnn = True", "enable_mkldnn = False")
    text = text.replace("args.enable_mkldnn = True", "args.enable_mkldnn = False")

    if text != original:
        backup = path.with_suffix(path.suffix + ".bak_hard_disable_onednn_v2")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        patched.append(path)

print("Patched files:")
for p in patched:
    print(p)

# Verify PaddleOCR files still compile.
for p in root.rglob("*.py"):
    py_compile.compile(str(p), doraise=True)

print("PaddleOCR Python files compile successfully.")
