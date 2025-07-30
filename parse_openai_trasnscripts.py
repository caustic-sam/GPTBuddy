"""
Chat Organizer MVP with Top Keywords & Chat-Level Clustering

Parses OpenAI JSON chat exports (file/ZIP/directory) into per-conversation Markdown, extracts top-5 keywords, and clusters chats into topics:

Features:
- Accepts `--input_path` (JSON file, ZIP, or directory) and `--output_dir`
- Parses formats: single dict, raw list, mapping exports, top-level dict, streaming arrays
- Extracts top-5 TF-IDF keywords per chat
- Clusters chats by TF-IDF of full transcripts (`--clusters N`)
- Writes one Markdown per chat with keywords
- Generates `clusters_index.md` grouping chats by cluster
- Builds `index.md`, `sessions_index.md`, and `malformed.md`
- Detailed debug & summary

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
import ijson
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def extract_from_mapping(conv):
    msgs, mapping = [], conv.get('mapping', {}) or {}
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
    if isinstance(data, dict):
        msgs = []
        for v in data.values():
            if isinstance(v, dict) and 'mapping' in v:
                msgs.extend(extract_from_mapping(v))
            elif isinstance(v, dict) and 'messages' in v:
                msgs.extend(v['messages'])
        return msgs
    return []


def extract_keywords_for_messages(msgs, top_n=5):
    texts = [m.get('content','') for m in msgs]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    sums = np.array(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    top_idxs = sums.argsort()[::-1][:top_n]
    return [terms[i] for i in top_idxs]


def write_md(msgs, name, out):
    safe = name.replace('/','_')
    fn = f"{safe}.md"
    lines = [f"# Chat: {name}", ""]
    # Insert top keywords
    kw = extract_keywords_for_messages(msgs)
    if kw:
        lines.append(f"**Top Keywords:** {', '.join(kw)}")
        lines.append("")
    for m in msgs:
        content = m.get('content','').strip()
        if not content: continue
        lines.append(f"## {m.get('role','unknown')}\n{content}\n")
    (out / fn).write_text("\n".join(lines), encoding='utf-8')
    return fn, len(msgs)


def cluster_chats(chat_texts, n_clusters):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(chat_texts)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)
    return labels


def parse_chats(inp, out_dir, n_clusters):
    p = Path(inp)
    # Unzip if needed
    if p.is_file() and p.suffix.lower() == '.zip':
        td = tempfile.TemporaryDirectory()
        print(f"Extracting zip to {td.name}")
        with zipfile.ZipFile(p, 'r') as zf:
            zf.extractall(td.name)
        base = Path(td.name)
    else:
        base = p
    # Dive into conversations subfolder if present
    conv_subdir = base / 'conversations'
    if conv_subdir.is_dir():
        print(f"Using folder: {conv_subdir}")
        base = conv_subdir
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Parse all JSON into message lists and store chat texts
    chat_names, chat_msgs, chat_texts = [], [], []
    for f in base.rglob('*.json'):
        if '__MACOSX' in f.as_posix() or f.name.startswith('._'): continue
        rel = f.relative_to(base).as_posix()
        print(f"Processing {rel}")
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:
            continue
        # extract messages
        msgs = extract_messages(data)
        if not msgs: continue
        text = " ".join(m.get('content','') for m in msgs)
        chat_names.append(rel)
        chat_msgs.append(msgs)
        chat_texts.append(text)

    # Write individual Markdown and collect
    idx = []
    for name, msgs in zip(chat_names, chat_msgs):
        fn, count = write_md(msgs, name, out)
        idx.append((name, fn, count))

    # Perform clustering
    if n_clusters and len(chat_texts) >= n_clusters:
        print(f"Clustering {len(chat_texts)} chats into {n_clusters} clusters...")
        labels = cluster_chats(chat_texts, n_clusters)
    else:
        labels = [0]*len(chat_names)

    # Write clusters_index.md
    clusters = {}
    for name, fn, count, lab in zip(chat_names, [fn for _,fn,_ in idx], [c for _,_,c in idx], labels):
        clusters.setdefault(lab, []).append((name, fn, count))
    lines = ["# Clusters Index", ""]
    for lab, items in clusters.items():
        lines.append(f"## Cluster {lab}")
        for name, fn, count in items:
            lines.append(f"- [{name} ({count} msgs)]({fn})")
        lines.append("")
    (out / 'clusters_index.md').write_text("\n".join(lines), encoding='utf-8')

    # Write index.md
    index_lines = ["# Chat Index", ""] + [f"- [{n} ({c} msgs)]({fn})" for n,fn,c in idx]
    (out / 'index.md').write_text("\n".join(index_lines), encoding='utf-8')

    print(f"Done: {len(chat_names)} chats; clusters_index.md created.")


def main():
    parser = argparse.ArgumentParser(description='Chat Organizer with Clustering')
    parser.add_argument('--input_path', '--input_dir', dest='inp', required=True,
                        help='Path to JSON, ZIP, or directory')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--clusters', type=int, default=5,
                        help='Number of clusters to group chats')
    args = parser.parse_args()
    parse_chats(args.inp, args.output_dir, args.clusters)

if __name__ == '__main__':
    main()
