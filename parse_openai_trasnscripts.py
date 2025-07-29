"""
Chat Organizer MVP (with Debugging)

This minimal script parses downloaded OpenAI JSON chat exports and generates basic Markdown outputs, while retaining malformed files for review and logging progress:
- Recursively finds all JSON files under input_dir
- Logs every JSON file discovered
- Extracts messages (role and content)
- Writes one Markdown file per source JSON, preserving relative path
- Generates an index.md linking to each Markdown file
- Collects any files that fail to load or have unexpected structure into malformed.md
- At end, prints summary counts of files processed and skipped

Usage:
  python chat_organizer.py \
    --input_dir ./downloads \
    --output_dir ./parsed_chats
"""

import argparse
import json
from pathlib import Path

def parse_chats(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Recursively find all JSON files
    json_files = list(input_path.rglob('*.json'))
    print(f"DEBUG: Found {len(json_files)} JSON files under '{input_dir}':")
    for jf in json_files:
        print(f"  - {jf.relative_to(input_path)}")

    index_entries = []
    skip_records = []  # list of (rel_path, error_message, raw_content)

    for json_file in json_files:
        rel = json_file.relative_to(input_path)
        rel_str = rel.as_posix()
        print(f"DEBUG: Processing {rel_str}")

        # Read raw text for potential retention
        try:
            raw = json_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Skipping {rel_str}: unable to read file ({e})")
            skip_records.append((rel_str, f"read_error: {e}", None))
            continue

        # Parse JSON
        try:
            data = json.loads(raw)
        except Exception as e:
            print(f"Skipping {rel_str}: JSON parse error ({e})")
            skip_records.append((rel_str, f"json_error: {e}", raw))
            continue

        # Determine messages list
        if isinstance(data, dict) and 'messages' in data:
            messages = data['messages']
        elif isinstance(data, list) and data and isinstance(data[0], dict) and 'messages' in data[0]:
            # Case: list of conversations
            # Flatten all messages across list
            messages = []
            for conv in data:
                msgs = conv.get('messages', [])
                messages.extend(msgs if isinstance(msgs, list) else [])
        elif isinstance(data, list):
            messages = data
        else:
            print(f"Skipping {rel_str}: unexpected structure")
            skip_records.append((rel_str, "structure_error", raw))
            continue

        # Write Markdown for valid chat
        md_name = rel.with_suffix('.md').as_posix().replace('/', '_')
        md_file = output_path / md_name
        md_lines = [f"# Chat: {rel_str}", ""]
        count_msgs = 0
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '').strip()
            if content:
                md_lines.append(f"## {role}\n{content}\n")
                count_msgs += 1
        if count_msgs == 0:
            print(f"WARNING: No messages found in {rel_str}")
        md_file.write_text("\n".join(md_lines), encoding='utf-8')
        index_entries.append((rel_str, md_name))
        print(f"DEBUG: Wrote {md_name} with {count_msgs} messages")

    # Write index.md
    idx_path = output_path / 'index.md'
    lines = ['# Chat Index', '']
    for rel_path, md_name in sorted(index_entries):
        lines.append(f"- [{rel_path}]({md_name})")
    idx_path.write_text("\n".join(lines), encoding='utf-8')

    # Write malformed.md if any
    if skip_records:
        malformed_path = output_path / 'malformed.md'
        m_lines = ['# Malformed Files', '']
        for rel_path, err, raw in skip_records:
            m_lines.append(f"## {rel_path}")
            m_lines.append(f"**Error:** {err}")
            if raw:
                m_lines.append('```json')
                snippet = raw if len(raw) < 10000 else raw[:10000] + '\n...'
                m_lines.append(snippet)
                m_lines.append('```')
            m_lines.append('')
        malformed_path.write_text("\n".join(m_lines), encoding='utf-8')
        print(f"Malformed files recorded in {malformed_path}")

    # Summary
    print(f"SUMMARY: Processed {len(index_entries)} files, skipped {len(skip_records)} malformed.")
    print(f"Markdown files generated in {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Chat Organizer MVP')
    parser.add_argument('--input_dir', required=True, help='Directory of JSON chat exports')
    parser.add_argument('--output_dir', required=True, help='Directory for Markdown output')
    args = parser.parse_args()
    parse_chats(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()
