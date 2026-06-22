from pathlib import Path


files_to_patch = [
    Path("localdoc/server.py"),
    Path("localdoc/report.py"),
]


old_block = '''def extract_supplier(text: str) -> str:
    if "ABC Foods LLC" in text:
        return "ABC Foods LLC"

    patterns = [
        r"Supplier legal name[:\\s]+([A-Z][A-Za-z0-9 &.,'-]+)",
        r"Insured supplier[:\\s]+([A-Z][A-Za-z0-9 &.,'-]+)",
        r"Supplier[:\\s]+([A-Z][A-Za-z0-9 &.,'-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            value = re.split(
                r"\\s+(DBA|Bank name|Policy|Document type|Coverage|Field|Primary contact|Email|Phone)\\b",
                value,
            )[0].strip()
            value = value.strip(". ")
            if value:
                return value

    return "Unknown supplier"
'''


old_block_server_variant = '''def extract_supplier(text: str) -> str:
    if "ABC Foods LLC" in text:
        return "ABC Foods LLC"

    patterns = [
        r"Supplier legal name[:\\s]+([A-Z][A-Za-z0-9 &.,'-]+)",
        r"Insured supplier[:\\s]+([A-Z][A-Za-z0-9 &.,'-]+)",
        r"Supplier[:\\s]+([A-Z][A-Za-z0-9 &.,'-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            value = re.split(
                r"\\s+(DBA|Bank name|Policy|Document type|Coverage|Field|Primary contact)\\b",
                value,
            )[0].strip()
            value = value.strip(". ")
            if value:
                return value

    return "Unknown supplier"
'''


new_block = '''def extract_supplier(text: str) -> str:
    """
    Extract supplier name from document text using generic evidence patterns.

    This intentionally avoids hard-coded supplier names so the system works on
    newly uploaded PDFs instead of only the sample packet.
    """
    patterns = [
        r"Supplier legal name\\s*[:\\-]?\\s*([^\\n\\r]+)",
        r"Insured supplier\\s*[:\\-]?\\s*([^\\n\\r]+)",
        r"Supplier name\\s*[:\\-]?\\s*([^\\n\\r]+)",
        r"Legal name\\s*[:\\-]?\\s*([^\\n\\r]+)",
        r"Vendor legal name\\s*[:\\-]?\\s*([^\\n\\r]+)",
        r"Vendor name\\s*[:\\-]?\\s*([^\\n\\r]+)",
        r"Company name\\s*[:\\-]?\\s*([^\\n\\r]+)",
    ]

    stop_words = [
        "DBA",
        "Bank name",
        "Policy",
        "Document type",
        "Coverage",
        "Field",
        "Primary contact",
        "Email",
        "Phone",
        "Address",
        "Tax",
        "W-9",
        "Certificate",
        "ACH",
        "Agreement",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if not match:
            continue

        value = match.group(1).strip()

        for stop_word in stop_words:
            value = re.split(r"\\s+" + re.escape(stop_word) + r"\\b", value, flags=re.IGNORECASE)[0].strip()

        value = value.strip(" .:-")

        if value and len(value) >= 2:
            return value

    # Fallback: look for common company suffixes near the beginning of the packet.
    first_part = text[:2500]

    company_match = re.search(
        r"([A-Z][A-Za-z0-9 &'.,-]{2,80}\\s+(LLC|Inc\\.?|Corporation|Corp\\.?|Ltd\\.?|Limited|Co\\.?|Company))",
        first_part,
    )

    if company_match:
        return company_match.group(1).strip(" .:-")

    return "Unknown supplier"
'''


for path in files_to_patch:
    if not path.exists():
        print(f"SKIPPED missing file: {path}")
        continue

    text = path.read_text(encoding="utf-8-sig")

    if old_block in text:
        text = text.replace(old_block, new_block)
        path.write_text(text, encoding="utf-8")
        print(f"Patched extract_supplier in {path}")
    elif old_block_server_variant in text:
        text = text.replace(old_block_server_variant, new_block)
        path.write_text(text, encoding="utf-8")
        print(f"Patched extract_supplier in {path}")
    elif "def extract_supplier(text: str) -> str:" in text:
        print(f"WARNING: Found extract_supplier in {path}, but exact block did not match. Manual review needed.")
    else:
        print(f"WARNING: No extract_supplier found in {path}")


print("Done.")
