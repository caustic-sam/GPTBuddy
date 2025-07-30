"""
Chat Organizer Full Script

Parses OpenAI JSON chat exports (file/ZIP/directory) into per-conversation Markdown,
extracts top-5 keywords, clusters chats, and generates both Markdown and HTML indices.

Usage:
  python chat_organizer_full.py \
    --input_path ./export.zip \
    --output_dir ./parsed_chats \
    [--clusters 5]

Dependencies:
  pip install ijson scikit-learn numpy
"""
import argparse
import json
import zipfile
import tempfile
import re
from pathlib import Path
from datetime import datetime
import ijson
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Helper to sanitize and truncate filenames
def sanitize_filename(s):
    """Clean and truncate filenames to a safe length"""
    s = s.replace("\n", " ").strip()
    s = re.sub(r'[\\/:*?"<>|]+', '', s)
    if len(s) > 50:
        s = s[:50].rstrip() + "..."
    return s

# Extract messages from mapping exports
def extract_from_mapping(conv):
    msgs = []
    mapping = conv.get('mapping', {}) or {}
    roots = [nid for nid,node in mapping.items() if not node.get('parent')]
    def dfs(node_id):
        node = mapping.get(node_id, {})
        md = node.get('message')
        if md:
            for part in md.get('content', {}).get('parts', []) or []:
                if isinstance(part, str) and part.strip():
                    msgs.append({'role': md.get('author',{}).get('role','unknown'), 'content': part})
        for cid in node.get('children', []) or []:
            dfs(cid)
    for root in roots:
        dfs(root)
    return msgs

# Extract messages for messages-format exports
def extract_messages(data):
    if isinstance(data, dict) and 'messages' in data:
        return data['messages']
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'role' in data[0] and 'content' in data[0]:
        return data
    return []

# Compute top-N TF-IDF keywords
def extract_keywords_for_messages(msgs, top_n=5):
    texts = [m.get('content','') for m in msgs]
    if not texts:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # Empty vocabulary (e.g., all stop words); no keywords to extract
        return []
    scores = np.asarray(X.sum(axis=0)).ravel()
    indices = np.argsort(scores)[::-1][:top_n]
    feature_names = np.array(vectorizer.get_feature_names_out())
    return feature_names[indices].tolist()

# Cluster chat texts
def cluster_chats(chat_texts, n_clusters):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(chat_texts)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    return km.fit_predict(X)

# Write a single chat to Markdown
def write_markdown(out_dir, filename, msgs):
    path = Path(out_dir) / filename
    lines = []
    for m in msgs:
        role = m.get('role','unknown')
        content = m.get('content','').strip()
        if not content:
            continue
        lines.append(f"## {role}\n{content}\n")
    path.write_text("\n".join(lines), encoding='utf-8')
    return filename, len(msgs)

# Main orchestration
def parse_chats(inp, out_dir, n_clusters):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []  # (base_name, msgs, text)
    base = Path(inp)
    if zipfile.is_zipfile(inp):
        z = zipfile.ZipFile(inp)
        tmp = Path(tempfile.mkdtemp())
        z.extractall(tmp)
        base = tmp

    for f in base.rglob('*.json'):
        if '__MACOSX' in f.as_posix() or f.name.startswith('._'): continue
        rel = f.relative_to(base).as_posix()
        print(f"Processing {rel}")
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"Skipping {rel}: {e}")
            continue

        # List-of-mapping exports
        if isinstance(data, list) and data and isinstance(data[0], dict) and 'mapping' in data[0]:
            for idx_conv, conv in enumerate(data):
                msgs = extract_from_mapping(conv)
                if msgs:
                    # Use conversation-level timestamp if available
                    create_ts = conv.get('create_time')
                    if create_ts is not None:
                        dt_obj = datetime.fromtimestamp(create_ts)
                    else:
                        dt_obj = datetime.fromtimestamp(f.stat().st_mtime)
                    dt = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                    uq = next((m['content'] for m in msgs if m.get('role')=='user'), '')
                    bn = f"{dt} - {uq}"
                    text = ' '.join(m.get('content','') for m in msgs)
                    records.append((bn, msgs, text))
            continue

        # Dict-of-convs exports
        msgs = []
        if isinstance(data, dict) and not isinstance(data.get('messages'), list) and any(
            isinstance(v, dict) and ('mapping' in v or 'messages' in v) for v in data.values()
        ):
            for conv in data.values():
                part = extract_from_mapping(conv) if 'mapping' in conv else conv.get('messages', [])
                msgs.extend(part)
            if msgs:
                # Use conversation-level timestamp if available
                create_ts = conv.get('create_time')
                if create_ts is not None:
                    dt_obj = datetime.fromtimestamp(create_ts)
                else:
                    dt_obj = datetime.fromtimestamp(f.stat().st_mtime)
                dt = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                uq = next((m['content'] for m in msgs if m.get('role')=='user'), '')
                bn = f"{dt} - {uq}"
                text = ' '.join(m.get('content','') for m in msgs)
                records.append((bn, msgs, text))
            continue

        # Single chat
        if isinstance(data, dict) and 'mapping' in data:
            msgs = extract_from_mapping(data)
        elif isinstance(data, dict) and 'messages' in data:
            msgs = data['messages']
        elif isinstance(data, list):
            msgs = data
        else:
            msgs = []

        if msgs:
            # Use conversation-level timestamp if available
            if isinstance(data, dict) and data.get('create_time') is not None:
                dt_obj = datetime.fromtimestamp(data['create_time'])
            else:
                dt_obj = datetime.fromtimestamp(f.stat().st_mtime)
            dt = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            uq = next((m['content'] for m in msgs if m.get('role')=='user'), '')
            bn = f"{dt} - {uq}"
            text = ' '.join(m.get('content','') for m in msgs)
            records.append((bn, msgs, text))

    # Generate Markdown indices
    index = []
    texts = [text for _, _, text in records]
    labels = cluster_chats(texts, n_clusters)
    for (bn, msgs, text), lab in zip(records, labels):
        fn, _ = write_markdown(out, sanitize_filename(bn) + '.md', msgs)
        index.append((bn, fn, len(msgs)))

    # Write clusters_index.md
    clusters = {}
    for (bn, fn, cnt), lab in zip(index, labels):
        clusters.setdefault(lab, []).append((bn, fn, cnt))
    clines = ['# Clusters Index', '']
    for lab, items in clusters.items():
        clines.append(f'## Cluster {lab}')
        for bn, fn, cnt in items:
            clines.append(f"- [{bn} ({cnt} msgs)]({fn})")
        clines.append('')
    (out/'clusters_index.md').write_text("\n".join(clines), encoding='utf-8')

    # Write index.md
    ilines = ['# Chat Index', '']
    for bn, fn, cnt in index:
        ilines.append(f"- [{bn} ({cnt} msgs)]({fn})")
    (out/'index.md').write_text("\n".join(ilines), encoding='utf-8')

    # Write HTML overview
    html = [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8"><title>Chat Index</title></head><body>',
        '<h1>Chat Index</h1>',
        '<table border="1" cellpadding="5">',
        '<tr><th>Date</th><th>Title</th><th>Keywords</th><th>Messages</th><th>Cluster</th></tr>'
    ]
    for (bn, fn, cnt), lab, (r_bn, r_msgs, _) in zip(index, labels, records):
        if ' - ' in bn:
            date, title = bn.split(' - ', 1)
        else:
            date, title = '', bn
        kws = extract_keywords_for_messages(r_msgs)
        html.append(
            f"<tr><td>{date}</td>"
            f"<td><a href='{fn}'>{sanitize_filename(title)}</a></td>"
            f"<td>{', '.join(kws)}</td>"
            f"<td>{cnt}</td>"
            f"<td>{lab}</td></tr>"
        )
    html.extend(['</table>', '</body></html>'])
    (out/'index.html').write_text("\n".join(html), encoding='utf-8')

    print(f"Done: {len(records)} chats; indices generated.")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Chat Organizer Full')
    parser.add_argument('--input_path','--input_dir',dest='inp',required=True,
                        help='Path to JSON file, ZIP, or directory')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for Markdown/HTML output')
    parser.add_argument('--clusters', type=int, default=5,
                        help='Number of clusters')
    args = parser.parse_args()
    parse_chats(args.inp, args.output_dir, args.clusters)
