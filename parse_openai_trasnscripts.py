"""
Chat Organizer MVP (File/Dir/ZIP Input, Streaming)

Parses OpenAI JSON chat exports—whether a single file, ZIP, or directory—into per-conversation Markdown:
- Accepts --input_path (file, ZIP, or directory)
- Recursively finds JSON files (or extracts ZIP)
- Detects real conversations folder if present
- Supports formats:
  • Single dict with 'messages'
  • Raw list of {role,content}
  • List-of-mapping exports
  • Top-level dict of conv objects
  • Monolithic array streaming via ijson
- Writes one Markdown per conversation
- Builds index.md and malformed.md
- Detailed debug & summary

Dependencies:
  pip install ijson

Usage:
  python chat_organizer.py \
    --input_path ./your_data.zip \
    --output_dir ./parsed_chats
"""
import argparse
import json
import zipfile
import tempfile
from pathlib import Path
import ijson

def extract_from_mapping(conv):
    msgs, mapping = [], conv.get('mapping', {}) or {}
    roots = [nid for nid,node in mapping.items() if not node.get('parent')]
    def dfs(node_id):
        node = mapping.get(node_id, {})
        md = node.get('message')
        if md:
            for part in md.get('content', {}).get('parts', []) or []:
                if isinstance(part, str) and part.strip():
                    msgs.append({'role': md.get('author', {}).get('role', 'unknown'), 'content': part})
        for cid in node.get('children', []) or []:
            dfs(cid)
    for r in roots:
        dfs(r)
    return msgs

def extract_messages(data):
    if isinstance(data, dict) and 'messages' in data:
        return data['messages']
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'role' in data[0] and 'content' in data[0]:
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'mapping' in data[0]:
        allm = []
        for conv in data:
            allm.extend(extract_from_mapping(conv))
        return allm
    if isinstance(data, dict):
        allm = []
        for k, v in data.items():
            if isinstance(v, dict) and 'mapping' in v:
                allm.extend(extract_from_mapping(v))
            elif isinstance(v, dict) and 'messages' in v:
                allm.extend(v['messages'])
        return allm
    return []

def write_md(msgs, name, out):
    lines = [f"# Chat: {name}", ""]
    count = 0
    for m in msgs:
        content = m.get('content', '').strip()
        if not content:
            continue
        role = m.get('role', 'unknown')
        lines.append(f"## {role}\n{content}\n")
        count += 1
    safe = name.replace('/', '_')
    fn = f"{safe}.md"
    (out / fn).write_text("\n".join(lines), encoding='utf-8')
    return fn, count

def parse_chats(inp, out_dir):
    p = Path(inp)
    # Extract ZIP if needed
    if p.is_file() and p.suffix.lower() == '.zip':
        td = tempfile.TemporaryDirectory()
        print(f"Extracting zip to {td.name}")
        with zipfile.ZipFile(p, 'r') as zf:
            zf.extractall(td.name)
        base = Path(td.name)
    else:
        base = p
    # Detect conversations/ subfolder
    conv_subdir = base / 'conversations'
    if conv_subdir.is_dir():
        print(f"Using conversations folder: {conv_subdir}")
        base = conv_subdir
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # Collect JSON files, filter out macOS artifacts
    files = [f for f in base.rglob('*.json') if '__MACOSX' not in f.as_posix() and not f.name.startswith('._')]
    print(f"Found {len(files)} JSON files under {base}")
    for f in files:
        print(f"Processing {f.relative_to(base)}")
    idx, skip = [], []
    for f in files:
        rel = f.relative_to(base).as_posix()
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:
            skip.append((rel, f"json_err:{e}", None))
            continue
        # Session index detection
        if isinstance(data, list) and data and all(isinstance(item, dict) and 'id' in item and 'title' in item for item in data):
            session_lines = [f"# Sessions from {rel}", ""]
            for item in data:
                session_lines.append(f"- **{item.get('title','')}** (ID: {item.get('id')}) created: {item.get('create_time')}")
            sess_fn = 'sessions_index.md'
            (out / sess_fn).write_text("\n".join(session_lines), encoding='utf-8')
            idx.append((f"sessions_{rel}", sess_fn, len(data)))
            continue
        msgs = extract_messages(data)
        # Streaming fallback for arrays
        if not msgs:
            try:
                for i, conv in enumerate(ijson.items(f.open('rb'), 'item')):
                    submsgs = extract_messages(conv)
                    if submsgs:
                        name = f"{rel}_part{i+1}"
                        fn, c = write_md(submsgs, name, out)
                        idx.append((name, fn, c))
                continue
            except Exception as e:
                skip.append((rel, f"stream_err:{e}", None))
                continue
        # Write normal messages
        fn, c = write_md(msgs, rel, out)
        idx.append((rel, fn, c))
    # Write index.md
    index_lines = ["# Chat Index", ""] + [f"- [{n} ({c} msgs)]({fn})" for n, fn, c in idx]
    (out / 'index.md').write_text("\n".join(index_lines), encoding='utf-8')
    # Write malformed.md
    if skip:
        mal_lines = ["# Malformed Files", ""]
        for rel, err, _ in skip:
            mal_lines += [f"## {rel}", f"**Error:** {err}", ""]
        (out / 'malformed.md').write_text("\n".join(mal_lines), encoding='utf-8')
        print(f"Malformed recorded in {out/'malformed.md'}")
    print(f"Done: {len(idx)} chats extracted; {len(skip)} skipped.")

def main():
    parser = argparse.ArgumentParser(description='Chat Organizer MVP')
    parser.add_argument('--input_path', '--input_dir', dest='inp', required=True,
                        help='Path to a JSON file, ZIP archive, or directory of JSON chats')
    parser.add_argument('--output_dir', required=True, help='Directory for Markdown output')
    args = parser.parse_args()
    parse_chats(args.inp, args.output_dir)

if __name__ == '__main__':
    main()
