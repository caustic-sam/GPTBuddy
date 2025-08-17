"""
Chat Organizer Full Script

Parses OpenAI JSON chat exports (file/ZIP/directory) into per-conversation Markdown,
extracts top-5 keywords, clusters chats, and generates both Markdown and HTML indices.

Usage:

  python buddy_with_html.py \
    --input_path ./export.zip \
    --output_dir ./parsed_chats \
    [--clusters 5]
     
  python buddy_with_html.py \
  --input_path ./export.zip \
  --output_dir ./parsed_chats \
  --clusters 5 \
  --export_html

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

# --------------------------------------------------------------------------------------
# HTML CLEANING UTILITY
# --------------------------------------------------------------------------------------
# Why this exists:
#   When we convert/chat-organize content, various tool artifacts can end up embedded in
#   the output as JSON blobs (e.g., results from automations or editor tools). These
#   fragments look like: {"result": "Successfully created text document ...", "textdoc_id": "..."}
#   or sometimes appear within fenced code blocks. These are noise for human readers.
#
# What this function does:
#   * Removes inline JSON artifacts that contain keys like "textdoc_id" or "jawbone_id".
#   * Removes lines that include known automation success messages (e.g., "Successfully created text document").
#   * Removes fenced JSON blocks that include those keys.
#   * Collapses excessive blank lines so the final HTML is tidy.
#
# Notes:
#   * We keep the logic conservative on purpose—only clearly noisy patterns are removed.
#   * If you discover more noise patterns later, add another regex below with a short
#     explanation so future-you understands why it exists.
# --------------------------------------------------------------------------------------
import re

def clean_html_content(html: str) -> str:
    """Remove tool/automation artifacts and other obvious noise from generated HTML.

    Parameters
    ----------
    html : str
        The raw HTML string we are about to write to disk.

    Returns
    -------
    str
        A cleaned HTML string with automation/tool noise removed and spacing normalized.
    """
    # ----------------------------------------------------------------------------------
    # 1) Strip single-line JSON artifacts that include automation IDs (textdoc_id, jawbone_id)
    #    Example to remove:
    #      {"result": "Successfully created text document ...", "textdoc_id": "abc123", ...}
    # ----------------------------------------------------------------------------------
    html = re.sub(r"\{[^{}]*?\btextdoc_id\b[^{}]*?\}", "", html, flags=re.DOTALL)
    html = re.sub(r"\{[^{}]*?\bjawbone_id\b[^{}]*?\}", "", html, flags=re.DOTALL)

    # ----------------------------------------------------------------------------------
    # 2) Remove explicit automation success lines embedded in HTML/text nodes
    # ----------------------------------------------------------------------------------
    html = re.sub(r"Successfully created text document[^<]*", "", html)

    # ----------------------------------------------------------------------------------
    # 3) Remove fenced JSON/code blocks that contain those IDs (defensive clean)
    #    This targets blocks like:
    #      ```json
    #      { ... "textdoc_id": "..." ... }
    #      ```
    # ----------------------------------------------------------------------------------
    html = re.sub(
        r"```(?:json)?\s*\{[\s\S]*?(?:textdoc_id|jawbone_id)[\s\S]*?\}\s*```",
        "",
        html,
        flags=re.IGNORECASE,
    )

    # ----------------------------------------------------------------------------------
    # 4) Collapse multiple blank lines / whitespace between tags for a neater output
    # ----------------------------------------------------------------------------------
    html = re.sub(r"\n\s*\n+", "\n\n", html)

    return html

# --------------------------------------------------------------------------------------
# MARKDOWN → HTML RENDERING HELPERS (OPTIONAL)
# --------------------------------------------------------------------------------------
# Goal:
#   Allow this script to emit a per-conversation HTML file alongside the Markdown,
#   so you can drop the cleaned HTML straight into WordPress or any static site.
#   We keep dependencies minimal by attempting to use the 'markdown' package if
#   available; otherwise, we fall back to a safe preformatted view.
# --------------------------------------------------------------------------------------

# A tiny, self-contained CSS theme (dark/light aware) so HTML looks professional
BASE_CSS = """
:root { --bg:#0b0f14; --fg:#e6edf3; --muted:#9fb0c0; --accent:#5fb3f7; --border:#1d2630; --code:#0f1720; --link:#7cc0ff; }
@media (prefers-color-scheme: light) { :root { --bg:#fff; --fg:#111827; --muted:#4b5563; --accent:#2563eb; --border:#e5e7eb; --code:#f8fafc; --link:#1d4ed8; } }
*{box-sizing:border-box} html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
main{max-width:900px;margin:3rem auto;padding:0 1.2rem}
h1,h2,h3,h4,h5,h6{font-weight:750;line-height:1.25;letter-spacing:-.01em;margin:1.8rem 0 .8rem}
h1{font-size:2.1rem} h2{font-size:1.6rem;border-bottom:1px solid var(--border);padding-bottom:.3rem}
h3{font-size:1.25rem;color:var(--accent)}
p,ul,ol{margin:1rem 0} ul,ol{padding-left:1.4rem} li+li{margin-top:.3rem}
a{color:var(--link);text-decoration:none} a:hover{text-decoration:underline}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace}
code{background:var(--code);border:1px solid var(--border);padding:.12rem .3rem;border-radius:6px;font-size:.95em}
pre{background:var(--code);border:1px solid var(--border);border-radius:12px;padding:1rem;overflow:auto}
pre code{background:transparent;border:0;padding:0}
blockquote{margin:1.2rem 0;padding:.8rem 1rem;border-left:4px solid var(--accent);background:rgba(95,179,247,.08);border-radius:6px}
table{width:100%;border-collapse:collapse;margin:1rem 0;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--border);padding:.6rem .5rem} th{text-align:left;background:rgba(95,179,247,.08)}
hr{border:0;border-top:1px solid var(--border);margin:2rem 0}
.header{margin-bottom:1rem}.title{font-size:2.3rem;font-weight:800}.subtitle{color:var(--muted);font-size:1rem}
"""


def md_to_html_body(md_text: str) -> str:
    """Convert Markdown text to HTML.

    Strategy:
      1) Try to use the 'markdown' package (if installed) for solid conversion
         with tables and fenced code.
      2) If the import fails, fall back to a safe <pre> rendering so nothing breaks.
    """
    try:
        import markdown  # optional dependency; we fail gracefully if missing
        return markdown.markdown(
            md_text,
            extensions=['extra','tables','fenced_code','sane_lists','admonition']
        )
    except Exception:
        # Conservative fallback: encode as preformatted text
        safe = (md_text.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'))
        return f"<pre><code>{safe}</code></pre>"


def wrap_html_document(title: str, body_html: str) -> str:
    """Wrap a body fragment in a minimal, responsive HTML document shell.

    We keep CSS inline so the file is fully portable (no extra assets).
    """
    return f"""<!doctype html>
<html lang=\"en\"><head>
<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{title}</title>
<style>{BASE_CSS}</style>
</head>
<body>
  <main>
    <header class=\"header\"><div class=\"title\">{title}</div>
      <div class=\"subtitle\">Converted from Markdown — cleaned</div></header>
    <article>
      {body_html}
    </article>
  </main>
</body>
</html>"""


def write_html_from_markdown(out_dir: Path, md_filename: str) -> str:
    """Read a Markdown file, render to HTML, clean it, and write alongside the source.

    Returns the HTML filename for reference.
    """
    md_path = out_dir / md_filename
    html_name = Path(md_filename).with_suffix('.html').name
    html_path = out_dir / html_name

    # Read the Markdown content
    md_text = md_path.read_text(encoding='utf-8')

    # Convert to HTML, then wrap in a document shell
    body_html = md_to_html_body(md_text)
    full_html = wrap_html_document(Path(md_filename).stem, body_html)

    # Clean the HTML to strip automation/tool noise
    cleaned = clean_html_content(full_html)
    html_path.write_text(cleaned, encoding='utf-8')
    return html_name

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

# Cluster chat texts with strong guards for empty/low-signal inputs.
def cluster_chats(chat_texts, n_clusters):
    """Cluster chat texts with strong guards for empty/low-signal inputs.

    Why this exists:
      Sklearn's TfidfVectorizer will raise `ValueError: empty vocabulary` when the
      documents are all blank/whitespace or reduce to stop words. That can happen
      with short/exported chats, or when messages are purely URLs/emojis.

    Strategy:
      * Pre-filter to identify which docs are non-empty after .strip().
      * Downsize n_clusters so it never exceeds the number of non-empty docs.
      * If TF-IDF still fails (e.g., all stop words), fall back to a single-cluster
        assignment (all zeros) so the pipeline continues gracefully.
      * Map predictions back to original indices; empty docs get cluster 0.
    """
    # Track which documents are non-empty after trimming whitespace
    non_empty_mask = [isinstance(t, str) and bool(t.strip()) for t in chat_texts]

    # Edge case: no non-empty documents -> return all zeros
    if not any(non_empty_mask):
        return np.zeros(len(chat_texts), dtype=int)

    # Build the list of texts that actually contain content
    non_empty_texts = [t for t, keep in zip(chat_texts, non_empty_mask) if keep]

    # Ensure n_clusters is within [1, len(non_empty_texts)]
    safe_k = max(1, min(int(n_clusters) if n_clusters else 1, len(non_empty_texts)))

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    try:
        X = vectorizer.fit_transform(non_empty_texts)
    except ValueError:
        # Empty vocabulary (e.g., all stop words) -> single cluster fallback
        labels_full = np.zeros(len(chat_texts), dtype=int)
        return labels_full

    # KMeans: set n_init explicitly to avoid sklearn warnings; use a fixed seed for reproducibility
    km = KMeans(n_clusters=safe_k, n_init=10, random_state=42)
    labels_non_empty = km.fit_predict(X)

    # Map back to the original positions (empty docs -> cluster 0)
    labels_full = np.zeros(len(chat_texts), dtype=int)
    j = 0
    for i, keep in enumerate(non_empty_mask):
        if keep:
            labels_full[i] = labels_non_empty[j]
            j += 1
        else:
            labels_full[i] = 0

    return labels_full

# Write a single chat to Markdown
def write_markdown(out_dir, filename, msgs):
    """Write a chat conversation to a Markdown file."""
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
def parse_chats(inp, out_dir, n_clusters, export_html=False):
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
                    dt = dt_obj.strftime('%Y-%m-%d')
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
                dt = dt_obj.strftime('%Y-%m-%d')
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
            dt = dt_obj.strftime('%Y-%m-%d')
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
        # Optionally emit a per-conversation HTML file next to the Markdown.
        # This is controlled by the --export_html flag so you can toggle it.
        if export_html:
            try:
                _ = write_html_from_markdown(out, fn)
            except Exception as e:
                # We log and keep going; a single conversion should not halt the whole run.
                print(f"[warn] HTML export failed for {fn}: {e}")
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

    # --------------------------------------------------------------------------------------
    # Finalize the HTML index:
    #   * Join the collected HTML fragments.
    #   * Run the result through our cleaner to strip tool/automation noise.
    #   * Write the cleaned HTML to disk.
    # --------------------------------------------------------------------------------------
    raw_index_html = "\n".join(html)
    cleaned_index_html = clean_html_content(raw_index_html)
    (out / 'index.html').write_text(cleaned_index_html, encoding='utf-8')

    print(f"Done: {len(records)} chats; indices generated.")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Chat Organizer Full')
    parser.add_argument('--input_path','--input_dir',dest='inp',required=True,
                        help='Path to JSON file, ZIP, or directory')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for Markdown/HTML output')
    parser.add_argument('--clusters', type=int, default=5,
                        help='Number of clusters')
    parser.add_argument('--export_html', action='store_true', help='Also export cleaned HTML per chat')
    args = parser.parse_args()
    parse_chats(args.inp, args.output_dir, args.clusters, export_html=args.export_html)
