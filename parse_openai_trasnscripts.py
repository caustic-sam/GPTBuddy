"""
Chat Organizer MVP

This minimal script parses downloaded OpenAI JSON chat exports and generates basic Markdown outputs:
- Recursively finds all JSON files under --input_dir
- Extracts messages (role and content)
- Writes one Markdown file per source JSON, preserving relative path
- Generates an index.md linking to each Markdown file

Usage:
  python chat_organizer.py \
    --input_dir ./downloads \
    --output_dir ./organized_md
"""

import os
import json
import argparse
from pathlib import Path


def parse_chats(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    index_entries = []

    for json_file in input_path.rglob('*.json'):
        try:
            data = json.loads(json_file.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"Failed to load {json_file}: {e}")
            continue

        # Determine messages list
        if isinstance(data, dict) and 'messages' in data:
            messages = data['messages']
        elif isinstance(data, list):
            messages = data
        else:
            print(f"Skipping {json_file}: unexpected structure")
            continue

        # Prepare Markdown file path
        rel = json_file.relative_to(input_path)
        md_name = rel.with_suffix('.md').as_posix().replace('/', '_')
        md_file = output_path / md_name
        md_content = [f"# {rel.as_posix()}\n"]

        # Write messages
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '').strip()
            if content:
                md_content.append(f"## {role}\n\n{content}\n")

        # Save Markdown file
        md_file.write_text('\n'.join(md_content), encoding='utf-8')
        index_entries.append((rel.as_posix(), md_name))

    # Write index.md
    idx_path = output_path / 'index.md'
    lines = ['# Chat Index', '']
    for rel_path, md_name in sorted(index_entries):
        lines.append(f"- [{rel_path}]({md_name})")
    idx_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Chat Organizer MVP')
    parser.add_argument('--input_dir', required=True, help='Directory of JSON chat exports')
    parser.add_argument('--output_dir', required=True, help='Directory for Markdown output')
    args = parser.parse_args()

    parse_chats(args.input_dir, args.output_dir)
    print(f"Markdown files generated in {args.output_dir}")


if __name__ == '__main__':
    main()
