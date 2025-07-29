"""
Chat Organizer Tool

This script processes downloaded OpenAI chat exports (JSON) and automatically categorizes messages into topic clusters, generating Markdown files with:  
- Cluster labels (custom or auto-generated)  
- Top keywords per cluster  
- AI-generated summaries per cluster (with local fallback)  
- Styled index.md with topic overview  

Features:
- Recursively load JSON chat exports from a directory tree (supports both list- and dict-formatted exports)
- Compute embeddings via OpenAI API (openai>=1.0.0) with fallback to local Sentence-Transformers
- Cluster messages into topics (KMeans)
- Extract top keywords via TF-IDF
- Optionally accept user-provided labels file
- Summarize each cluster using ChatCompletion, falling back to keyword-based summary on any error
- Output one Markdown per topic and a styled index.md

Usage:
  1. Set your OpenAI API key (optional if using local embeddings):
     export OPENAI_API_KEY=your_api_key_here

  2. Install dependencies:
     pip install openai scikit-learn tqdm markdown sentence-transformers

  3. (Optional) Create a CSV labels file with two columns: cluster,label

  4. Run the script:
     python chat_organizer.py \
       --input_dir ./downloads \
       --output_dir ./organized_chats \
       --clusters 5 \
       [--labels_file cluster_labels.csv]

This will process your chats, cluster them, and generate topic files with summaries or keyword fallbacks.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import openai
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import csv

# Load a local embedding model for fallback
_local_model = SentenceTransformer('all-MiniLM-L6-v2')


def load_messages(input_dir):
    """Recursively load messages from all JSON files under input_dir."""
    messages, sources = [], []
    for filepath in Path(input_dir).rglob('*.json'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {filepath}: {e}")
            continue

        # Handle both dict and list chat exports
        if isinstance(data, dict) and 'messages' in data:
            msg_list = data['messages']
        elif isinstance(data, list):
            msg_list = data
        else:
            print(f"Warning: unexpected JSON structure in {filepath}")
            continue

        for idx, msg in enumerate(msg_list):
            content = msg.get('content', '').strip()
            if content:
                messages.append(content)
                role = msg.get('role', 'unknown')
                sources.append(f"{filepath.relative_to(input_dir)} [msg:{idx}][{role}]")
    return messages, sources


def get_embeddings(texts, batch_size=20):
    """Compute embeddings via OpenAI (if available) with fallback to local Sentence-Transformer."""
    openai.api_key = os.getenv('OPENAI_API_KEY')
    all_embeds = []
    use_openai = bool(openai.api_key)
    for i in tqdm(range(0, len(texts), batch_size), desc='Embedding'):
        batch = texts[i:i + batch_size]
        if use_openai:
            try:
                resp = openai.embeddings.create(
                    model='text-embedding-ada-002',
                    input=batch
                )
                all_embeds.extend([item['embedding'] for item in resp['data']])
                continue
            except Exception as e:
                print(f"OpenAI embedding failed ({e}), falling back to local model...")
                use_openai = False
        # Local fallback
        local_embeds = _local_model.encode(batch, convert_to_numpy=True)
        all_embeds.extend(local_embeds)
    return np.array(all_embeds)


def cluster_messages(embeddings, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    return km.fit_predict(embeddings)


def extract_keywords(messages, labels, top_n=10):
    vect = TfidfVectorizer(stop_words='english', max_features=10000)
    X = vect.fit_transform(messages)
    terms = np.array(vect.get_feature_names_out())
    keywords = {}
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0]
        sums = X[idxs].sum(axis=0)
        top_idx = np.argsort(np.array(sums).ravel())[::-1][:top_n]
        keywords[c] = terms[top_idx].tolist()
    return keywords


def load_labels(labels_file):
    labels_map = {}
    with open(labels_file, newline='', encoding='utf-8') as csvf:
        reader = csv.reader(csvf)
        for row in reader:
            if len(row) >= 2 and row[0].isdigit():
                labels_map[int(row[0])] = row[1]
    return labels_map


def summarize_cluster(messages, model='gpt-3.5-turbo', max_tokens=150):
    """Summarize messages for a cluster using ChatCompletion."""
    prompt = (
        "Summarize the following messages into a concise paragraph of key points:\n\n"
        + "\n---\n".join(messages)
    )
    resp = openai.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': prompt}],
        max_tokens=max_tokens,
        temperature=0.3
    )
    return resp['choices'][0]['message']['content'].strip()


def write_markdown(messages, sources, labels, keywords, summaries, cluster_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    index_lines = [
        '# Chat Topics Overview',
        '',
        '| Topic | Label | Keywords | Summary | #Messages |',
        '|-------|-------|----------|---------|-----------|'
    ]

    for c in sorted(set(labels)):
        label = cluster_labels.get(c, f"Topic {c+1}")
        kw = keywords[c][:5]
        summary = summaries.get(c, 'No summary available.').replace('\n',' ')
        count = int((labels==c).sum())
        filename = f'topic_{c+1}.md'
        index_lines.append(
            f"| [{c+1}]({filename}) | {label} | {', '.join(kw)} | {summary[:100]}... | {count} |"
        )

        with open(Path(output_dir)/filename, 'w', encoding='utf-8') as f:
            f.write(f"# {label}\n\n")
            f.write(f"**Top Keywords:** {', '.join(keywords[c])}\n\n")
            f.write(f"**Summary:** {summaries.get(c,'')}\n\n")
            for msg, src, lbl in zip(messages, sources, labels):
                if lbl == c:
                    f.write(f"- **{src}**  \n  {msg}\n\n")

    with open(Path(output_dir)/'index.md', 'w', encoding='utf-8') as idx:
        idx.write('\n'.join(index_lines))


def main():
    parser = argparse.ArgumentParser(
        description='Organize OpenAI chats into enhanced topic Markdown files'
    )
    parser.add_argument('--input_dir', required=True,
                        help='Root directory to recursively search for chat JSON files')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--clusters', type=int, default=5)
    parser.add_argument('--labels_file', help='CSV mapping cluster->label')
    args = parser.parse_args()

    print('Loading messages...')
    messages, sources = load_messages(args.input_dir)

    print('Computing embeddings...')
    embeds = get_embeddings(messages)

    print(f'Clustering into {args.clusters} topics...')
    labels = cluster_messages(embeds, args.clusters)

    print('Extracting keywords...')
    keywords = extract_keywords(messages, labels)

    # Load or generate cluster labels
    cluster_labels = load_labels(args.labels_file) if args.labels_file else {c: ' / '.join(keywords[c][:3]) for c in keywords}

    print('Summarizing clusters...')
    summaries = {}
    for c in sorted(set(labels)):
        msgs = [m for m, lbl in zip(messages, labels) if lbl == c]
        try:
            summaries[c] = summarize_cluster(msgs)
        except Exception as e:
            print(f"Summarization failed ({e}), using keyword fallback...")
            summaries[c] = 'Keywords: ' + ', '.join(keywords[c][:5])

    print('Writing Markdown...')
    write_markdown(messages, sources, labels, keywords, summaries, cluster_labels, args.output_dir)

    print('Done! Organized chats are in', args.output_dir)

if __name__ == '__main__':
    main()
