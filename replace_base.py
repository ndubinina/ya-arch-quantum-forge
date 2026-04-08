import json
import os
import re

terms_file = "terms_map.json"
kb_folder = "knowledge_base"

with open(terms_file, "r", encoding="utf-8") as f:
    terms_map = json.load(f)

def replace_match(match, replacement):
    word = match.group()
    if word[0].isupper():
        return replacement.capitalize()
    else:
        return replacement.lower()

for filename in os.listdir(kb_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(kb_folder, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        for old, new in terms_map.items():
            pattern = re.compile(re.escape(old), re.IGNORECASE)
            content = pattern.sub(lambda m: replace_match(m, new), content)

        # Сохраняем обратно
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)