"""
Chat Organizer MVP with Named Files by Date & Initial Question

Parses OpenAI JSON chat exports into per-conversation Markdown named as `YY-MM-DD - Initial Question`, extracts top-5 keywords, and clusters chats.

Features:
- Accepts `--input_path` (JSON file, ZIP, or directory) and `--output_dir`
- Splits mapping-exports, dict-of-convs, list-of-convs into individual chats
- Names each Markdown file using the conversation’s first user message and file modification date: `yy-mm-dd - question.md`
- Extracts top-5 TF-IDF keywords per chat
- Clusters chats by TF-IDF of full transcripts (`--clusters N`)
- Generates `clusters_index.md`, `index.md`, and `malformed.md`

Dependencies:
  pip install ijson scikit-learn numpy

Usage:
  python chat_organizer.py \
    --input_path ./export.zip \
    --output_dir ./parsed_chats \
    [--clusters 5]
"""
import argparse
import json
import zipfile
import tempfile
from pathlib import Path
import re
import ijson
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Helper to sanitize filenames
# Helper to sanitize and truncate filenames
# Helper to sanitize and truncate filenames
def sanitize_filename(s):
    """Clean and truncate filenames to a safe length"""
    # Replace newline characters with spaces and strip whitespace
    s = s.replace("", " ").strip()
    # Remove invalid filesystem characters
    s = re.sub(r'[\/:*?"<>|]+', '', s)
    # Truncate to avoid filesystem limits
    max_len = 50
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "..."
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
                if isinstance(part,str) and part.strip():
                    msgs.append({'role': md.get('author',{}).get('role','unknown'), 'content': part})
        for cid in node.get('children', []) or []:
            dfs(cid)
    for r in roots: dfs(r)
    return msgs

# Extract messages for all formats
def extract_messages(data):
    if isinstance(data, dict) and 'messages' in data:
        return data['messages']
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'role' in data[0] and 'content' in data[0]:
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'mapping' in data[0]:
        out = []
        for conv in data:
            out.extend(extract_from_mapping(conv))
        return out
    if isinstance(data, dict) and not isinstance(data.get('messages'), list):
        out = []
        for v in data.values():
            if isinstance(v, dict):
                if 'mapping' in v:
                    out.extend(extract_from_mapping(v))
                elif 'messages' in v:
                    out.extend(v['messages'])
        return out
    return []

# Top-5 TF-IDF keywords
def extract_keywords_for_messages(msgs, top_n=5):
    """Compute top-N TF-IDF keywords, returning empty list on failure"""
    texts = [m.get('content','') for m in msgs]
    if not texts:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # all content was stopwords or empty
        return []
    sums = np.array(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    top_idxs = sums.argsort()[::-1][:top_n]
    return [terms[i] for i in top_idxs]

# Write Markdown file with given filename base
def write_md_file(msgs, base_name, out_dir):
    fn_safe = sanitize_filename(base_name) + '.md'
    path = out_dir / fn_safe
    lines = [f"# Chat: {base_name}", ""]
    # Keywords
    kw = extract_keywords_for_messages(msgs)
    if kw:
        lines.append(f"**Top Keywords:** {', '.join(kw)}")
        lines.append("")
    # Messages
    for m in msgs:
        c = m.get('content','').strip()
        if not c: continue
        lines.append(f"## {m.get('role','unknown')}\n{c}\n")
    path.write_text("\n".join(lines), encoding='utf-8')
    return fn_safe, len(msgs)

# Cluster chat texts
def cluster_chats(chat_texts, n_clusters):
    vect = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vect.fit_transform(chat_texts)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    return km.fit_predict(X)

# Main processing
def parse_chats(inp, out_dir, n_clusters):
    p = Path(inp)
    # Unzip
    if p.is_file() and p.suffix.lower()=='.zip':
        td = tempfile.TemporaryDirectory()
        print(f"Extracting zip to {td.name}")
        with zipfile.ZipFile(p,'r') as zf:
            zf.extractall(td.name)
        base = Path(td.name)
    else:
        base = p
    # Dive into conversations/
    sub = base / 'conversations'
    if sub.is_dir(): base = sub
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    records = []  # tuples of (base_name, msgs, text)

    for f in base.rglob('*.json'):
        if '__MACOSX' in f.as_posix() or f.name.startswith('._'): continue
        rel = f.relative_to(base).as_posix()
        print(f"Processing {rel}")
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"Skipping {rel}: {e}")
            continue
        # Handle mapping lists
        if isinstance(data, list) and data and isinstance(data[0], dict) and 'mapping' in data[0]:
            for idx_conv, conv in enumerate(data):
                msgs = extract_from_mapping(conv)
                if msgs:
                    # get date
                    dt = datetime.fromtimestamp(f.stat().st_mtime).strftime('%y-%m-%d')
                    # first user question
                    uq = next((m['content'] for m in msgs if m.get('role')=='user'), '')
                    base_name = f"{dt} - {uq}"
                    text = ' '.join(m.get('content','') for m in msgs)
                    records.append((base_name, msgs, text))
            continue
        # Dict-of-convs
        if isinstance(data, dict) and not isinstance(data.get('messages'), list) and any(
            isinstance(v, dict) and ('mapping' in v or 'messages' in v) for v in data.values()
        ):
            for key, conv in data.items():
                msgs = extract_from_mapping(conv) if 'mapping' in conv else conv.get('messages', [])
                if msgs:
                    dt = datetime.fromtimestamp(f.stat().st_mtime).strftime('%y-%m-%d')
                    uq = next((m['content'] for m in msgs if m.get('role')=='user'), '')
                    base_name = f"{dt} - {uq}"
                    text = ' '.join(m.get('content','') for m in msgs)
                    records.append((base_name, msgs, text))
            continue
        # Single chat
        msgs = extract_from_mapping(data) if isinstance(data, dict) and 'mapping' in data else data.get('messages') if isinstance(data, dict) else data if isinstance(data,list) else []
        if msgs:
            dt = datetime.fromtimestamp(f.stat().st_mtime).strftime('%y-%m-%d')
            uq = next((m['content'] for m in msgs if m.get('role')=='user'), '')
            base_name = f"{dt} - {uq}"
            text = ' '.join(m.get('content','') for m in msgs)
            records.append((base_name, msgs, text))

    # Write Markdown and collect index
    index = []
    chat_texts = []
    for base_name, msgs, text in records:
        fn, cnt = write_md_file(msgs, base_name, out)
        index.append((base_name, fn, cnt))
        chat_texts.append(text)

    # Clustering
    labels = [0]*len(records)
    if n_clusters and len(records)>=n_clusters:
        print(f"Clustering {len(records)} chats into {n_clusters} clusters...")
        labels = cluster_chats(chat_texts, n_clusters)

    # clusters_index.md
    clusters = {}
    for (base_name, fn, cnt), lab in zip(index, labels):
        clusters.setdefault(lab, []).append((base_name, fn, cnt))
    lines = ['# Clusters Index','']
    for lab, items in clusters.items():
        lines.append(f'## Cluster {lab}')
        for bn,fn,c in items:
            lines.append(f"- [{bn} ({c} msgs)]({fn})")
        lines.append('')
    (out/'clusters_index.md').write_text("\n".join(lines), encoding='utf-8')

    # index.md
    lines = ['# Chat Index','']
    for bn,fn,c in index:
        lines.append(f"- [{bn} ({c} msgs)]({fn})")
    (out/'index.md').write_text("\n".join(lines), encoding='utf-8')

    print(f"Done: {len(records)} chats; clusters_index.md created.")

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Chat Organizer with Named Files')
    p.add_argument('--input_path','--input_dir',dest='inp',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--clusters',type=int,default=5)
    a=p.parse_args()
    parse_chats(a.inp,a.output_dir,a.clusters)
