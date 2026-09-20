import re
from pathlib import Path

content = Path(r"C:\Users\Rupika\.gemini\antigravity-ide\brain\3e0415f4-402a-4a38-84cd-2d9e3aeb92dc\.system_generated\steps\267\content.md").read_text(encoding="utf-8", errors="ignore")

csv_matches = re.findall(r'[\w\-\.]+\.csv', content, re.IGNORECASE)
print("CSV matches:", set(csv_matches))

# Look for drive file IDs
# Google Drive links often have format: https://drive.google.com/file/d/<ID>/view
id_matches = re.findall(r'https://drive\.google\.com/file/d/([a-zA-Z0-9_\-]+)', content)
print("File links:", set(id_matches))

# Also search for "train" or "test" strings in JSON payloads
matches = re.findall(r'("[^"]*(?:train|test|triage)[^"]*")', content, re.IGNORECASE)
print("Keyword matches:", set(matches[:30]))
