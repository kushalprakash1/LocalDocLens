import re
from pathlib import Path

cli_path = Path("localdoc/cli.py")
text = cli_path.read_text(encoding="utf-8-sig")

report_command = '''

@app.command()
def report(
    output_dir: str = "artifacts",
    file: str = "",
):
    """
    Generate a structured supplier compliance report from the indexed packet.

    Use --file when multiple PDFs are indexed so evidence from different suppliers
    does not get mixed into one report.
    """
    from localdoc.report import build_supplier_report

    selected_file = file.strip() or None

    try:
        result = build_supplier_report(output_dir=output_dir, file_name=selected_file)
    except Exception as exc:
        print("")
        print("Could not generate report.")
        print(str(exc))
        raise typer.Exit(code=1)

    print("")
    print("LocalDocLens supplier compliance report generated.")
    print(f"Supplier: {result['supplier']}")
    print(f"Source File: {result['source_file']}")
    print(f"Overall Risk: {result['overall_risk']}")
    print(f"Decision: {result['decision']}")
    print("")
    print("Saved:")
    print(result["output_files"]["markdown"])
    print(result["output_files"]["json"])
'''

inspect_command = '''

@app.command()
def inspect(
    output_dir: str = "artifacts",
):
    """
    Inspect indexed PDFs, pages, chunks, OCR confidence, and answer-cache counts.
    """
    from localdoc.inspect import inspect_index

    result = inspect_index(output_dir=output_dir)

    print("")
    print("LocalDocLens index inspection")
    print(f"Total files: {result['total_files']}")
    print(f"Total chunks: {result['total_chunks']}")
    print("")

    for item in result["files"]:
        print(f"- {item['file_name']}")
        print(f"  pages: {item['pages']}")
        print(f"  chunks: {item['chunks']}")
        print(f"  extraction methods: {item['extraction_methods']}")
        print(f"  avg OCR confidence: {item['avg_ocr_confidence']}")
        print("")

    cache = result["cache"]
    print("Cache:")
    print(f"  total cached answers: {cache['total_cached_answers']}")
    print(f"  verified cached answers: {cache['verified_cached_answers']}")
    print(f"  total cache hits: {cache['total_cache_hits']}")
    print("")
    print("Saved:")
    print(result["output_file"])
'''

def replace_or_insert_command(text: str, command_name: str, command_text: str) -> str:
    pattern = (
        r'\\n@app\\.command\\(\\)\\n'
        r'def ' + re.escape(command_name) + r'\\([\\s\\S]*?'
        r'(?=\\n\\n@app\\.command\\(\\)|\\n\\nif __name__ == "__main__":|\\Z)'
    )

    if re.search(pattern, text):
        return re.sub(pattern, "\\n" + command_text.strip("\\n"), text, count=1)

    marker = 'if __name__ == "__main__":'

    if marker in text:
        return text.replace(marker, command_text + "\\n\\n" + marker)

    return text.rstrip() + "\\n" + command_text + "\\n"


if "import typer" not in text:
    text = "import typer\\n" + text

text = replace_or_insert_command(text, "report", report_command)
text = replace_or_insert_command(text, "inspect", inspect_command)

cli_path.write_text(text, encoding="utf-8")

print("Updated localdoc report command.")
print("Added/updated localdoc inspect command.")
