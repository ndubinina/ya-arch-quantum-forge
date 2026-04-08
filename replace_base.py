import json
import os

terms_file = "terms_map.json"
kb_folder = "knowledge_base"

with open(terms_file, "r", encoding="utf-8") as f:
    terms_map = json.load(f)

for filename in os.listdir(kb_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(kb_folder, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        for old, new in terms_map.items():
            content = content.replace(old, new)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)