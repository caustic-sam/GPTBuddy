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
    for r in roots:
        dfs(r)
    return msgs

# Extract messages for various formats
def extract_messages(data):
    if isinstance(data, dict) and 'messages' in data:
        return data['messages']
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'role' in data[0] and 'content' in data[0]:
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'mapping' in data[0]:
        msgs = []
        for conv in data:
            msgs.extend(extract_from_mapping(conv))
        return msgs
    if isinstance(data, dict) and not isinstance(data.get('messages'), list):
        msgs = []
        for v in data.values():
            if isinstance(v, dict):
                if 'mapping' in v:
                    msgs.extend(extract_from_mapping(v))
                elif 'messages' in v:
                    msgs.extend(v['messages'])
        return msgs
    return []

# Compute top-N TF-IDF keywords
def extract_keywords_for_messages(msgs, top_n=5):
    texts = [m.get('content','') for m in msgs]
    if not texts:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    sums = np.array(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    top_idxs = sums.argsort()[::-1][:top_n]
    return [terms[i] for i in top_idxs]

# Write Markdown file for a chat
def write_md_file(msgs, base_name, out_dir):
    fn = sanitize_filename(base_name) + '.md'
    path = out_dir / fn
    lines = [f"# Chat: {base_name}", ""]
    kw = extract_keywords_for_messages(msgs)
    if kw:
        lines.append(f"**Top Keywords:** {', '.join(kw)}")
        lines.append("")
    for m in msgs:
        c = m.get('content','').strip()
        if not c: continue
        lines.append(f"## {m.get('role','unknown')}\n{c}\n")
    path.write_text("\n".join(lines), encoding='utf-8')
    return fn, len(msgs)

# Cluster chat texts
def cluster_chats(chat_texts, n_clusters):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(chat_texts)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    return km.fit_predict(X)

# Main orchestration
def parse_chats(inp, out_dir, n_clusters):
    p = Path(inp)
    # Unzip if archive
    if p.is_file() and p.suffix.lower() == '.zip':
        tmp = tempfile.TemporaryDirectory()
        print(f"Extracting ZIP to {tmp.name}")
        with zipfile.ZipFile(p,'r') as zf:
            zf.extractall(tmp.name)
        base = Path(tmp.name)
    else:
        base = p
    # Dive into conversations/ subfolder
    sub = base / 'conversations'
    if sub.is_dir():
        print(f"Using folder: {sub}")
        base = sub
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    records = []  # (base_name, msgs, text)
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
                    dt = datetime.fromtimestamp(f.stat().st_mtime).strftime('%y-%m-%d')
                    uq = next((m['content'] for m in msgs if m.get('role')=='user'), '')
                    bn = f"{dt} - {uq}"
                    text = ' '.join(m.get('content','') for m in msgs)
                    records.append((bn, msgs, text))
            continue
        # Dict-of-convs exports
        if isinstance(data, dict) and not isinstance(data.get('messages'), list) and any(
            isinstance(v, dict) and ('mapping' in v or 'messages' in v) for v in data.values()
        ):
            for key, conv in data.items():
                msgs = extract_from_mapping(conv) if 'mapping' in conv else conv.get('messages', [])
                if msgs:
                    dt = datetime.fromtimestamp(f.stat().st_mtime).strftime('%y-%m-%d')
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
            dt = datetime.fromtimestamp(f.stat().st_mtime).strftime('%y-%m-%d')
            uq = next((m['content'] for m in msgs if m.get('role')=='user'), '')
            bn = f"{dt} - {uq}"
            text = ' '.join(m.get('content','') for m in msgs)
            records.append((bn, msgs, text))

    # Write Markdown and gather texts
    index = []
    chat_texts = []
    for bn, msgs, text in records:
        fn, cnt = write_md_file(msgs, bn, out)
        index.append((bn, fn, cnt))
        chat_texts.append(text)

    # Cluster chats
    labels = [0]*len(records)
    if n_clusters and len(records) >= n_clusters:
        print(f"Clustering {len(records)} chats into {n_clusters} clusters...")
        labels = cluster_chats(chat_texts, n_clusters)

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
