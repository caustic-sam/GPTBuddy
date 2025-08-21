"""
Chat Organizer Full Script

Parses OpenAI JSON chat exports (file/ZIP/directory) into per-conversation Markdown,
extracts top-5 keywords, clusters chats, and generates both Markdown and HTML indices.

Usage:

***./parsed_chats can be deleted prior to running

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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Simple HTML escape for safe text insertion into attributes/nodes
import html as _py_html

def html_escape(s: str) -> str:
    return _py_html.escape(s, quote=True) if isinstance(s, str) else str(s)

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

# CSS for index.html page (zebra rows, badges, clean fonts/colors)
INDEX_CSS = """
:root { --bg:#0b0f14; --fg:#e6edf3; --muted:#9fb0c0; --accent:#7dd3fc; --accent2:#a78bfa; --border:#1d2630; }
@media (prefers-color-scheme: light) { :root { --bg:#ffffff; --fg:#0f172a; --muted:#475569; --accent:#0284c7; --accent2:#7c3aed; --border:#e2e8f0; } }
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
main{max-width:1100px;margin:3rem auto;padding:0 1rem}
header .title{font-weight:800;font-size:2.2rem;letter-spacing:-.01em;margin-bottom:.25rem}
header .subtitle{color:var(--muted)}
.table-wrap{margin-top:1.25rem;border:1px solid var(--border);border-radius:12px;overflow:hidden}
table{width:100%;border-collapse:collapse}
thead th{background:linear-gradient(90deg,var(--accent),var(--accent2));color:#001018;padding:.9rem .8rem;text-align:left;font-weight:800}
tbody td{border-top:1px solid var(--border);padding:.7rem .8rem;vertical-align:top}
tbody tr:nth-child(odd){background:rgba(125,211,252,.06)}
tbody tr:hover{background:rgba(167,139,250,.10)}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.badge{display:inline-block;padding:.18rem .5rem;border:1px solid var(--border);border-radius:999px;font-size:.8rem;color:var(--muted)}
kbd{background:#0f172a;color:#e2e8f0;border:1px solid #1f2937;border-radius:6px;padding:.1rem .35rem;font-size:.8rem}
.toolbar{display:flex;gap:.5rem;align-items:center;margin:.8rem 0 1rem}
input[type="search"]{width:100%;max-width:520px;padding:.55rem .7rem;border:1px solid var(--border);border-radius:10px;background:transparent;color:var(--fg)}
input[type="search"]::placeholder{color:var(--muted)}
/* Sticky header + shadow when scrolled */
thead th{position:sticky; top:0; z-index:2}
.thead-shadow thead th{box-shadow:0 2px 0 rgba(0,0,0,.1)}

/* Smooth transitions for hover/focus */
.table-wrap, thead th, tbody tr{transition:background-color .15s ease, box-shadow .15s ease}

/* Copy-link button */
.copy{cursor:pointer; border:1px solid var(--border); border-radius:8px; padding:.2rem .45rem; font-size:.85rem; background:transparent; color:var(--muted)}
.copy:hover{color:var(--fg); border-color:var(--accent)}
.copy:active{transform:translateY(1px)}
.copy[aria-busy="true"]{opacity:.7}

/* Mark (highlight) */
mark{background:rgba(250,204,21,.35); color:inherit; padding:0 .15rem; border-radius:4px}
/* Pagination */
.pager{display:flex;gap:.5rem;align-items:center;justify-content:flex-end;margin:.6rem 0}
.pager .pages{display:inline-flex;gap:.25rem;align-items:center}
.pager .pg{cursor:pointer;border:1px solid var(--border);border-radius:8px;padding:.25rem .55rem;background:transparent;color:var(--muted)}
.pager .pg[disabled]{opacity:.5;cursor:not-allowed}
.pager .pg:hover:not([disabled]){color:var(--fg);border-color:var(--accent)}
.pager .pp{margin-left:.5rem;border:1px solid var(--border);border-radius:8px;background:transparent;color:var(--fg);padding:.25rem .4rem}
/* Sort indicators */
thead th[data-sort]{cursor:pointer; position:relative}
thead th[data-sort].sorted-asc::after{content:" ▲"; font-size:.8em; color:var(--muted)}
thead th[data-sort].sorted-desc::after{content:" ▼"; font-size:.8em; color:var(--muted)}
thead th[data-sort].active{background:rgba(125,211,252,.18)}
"""


def md_to_html_body(md_text: str) -> str:
    """Convert Markdown text to HTML.

    Strategy:
      1) Try to use the 'markdown' package (if installed) for solid conversion
         with tables and fenced code.
      2) If the import fails, fall back to a safe <pre> rendering so nothing breaks.
    """
    try:
        import markdown  # type: ignore # optional dependency; we fail gracefully if missing
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
    scores = np.asarray(X.sum(axis=0)).ravel() # type: ignore
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
def write_markdown(out_dir, filename, msgs, title=None):
    """Write a chat conversation to a Markdown file.

    Parameters
    ----------
    out_dir : str | Path
        Destination directory
    filename : str
        File name to write (already sanitized)
    msgs : list[dict]
        Conversation messages
    title : str | None
        Optional human-friendly title (we'll write it as an H1 at the top when provided)
    """
    path = Path(out_dir) / filename
    lines = []

    # If we have a full title (e.g., date + full initial user query), write it once at the top
    if title and str(title).strip():
        lines.append(f"# {title}\n")

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
        fn, _ = write_markdown(out, sanitize_filename(bn) + '.md', msgs, title=bn)
        html_fn = None
        # Optionally emit a per-conversation HTML file next to the Markdown.
        # This is controlled by the --export_html flag so you can toggle it.
        if export_html:
            try:
                html_fn = write_html_from_markdown(out, fn)  # returns the HTML filename
            except Exception as e:
                # We log and keep going; a single conversion should not halt the whole run.
                print(f"[warn] HTML export failed for {fn}: {e}")
        index.append((bn, fn, len(msgs), html_fn))

    # Write clusters_index.md
    clusters = {}
    for (bn, fn, cnt, _html_fn), lab in zip(index, labels):
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
    for bn, fn, cnt, _html_fn in index:
        ilines.append(f"- [{bn} ({cnt} msgs)]({fn})")
    (out/'index.md').write_text("\n".join(ilines), encoding='utf-8')

    # Write HTML overview
    html = [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Chat Index</title>',
        f'<style>{INDEX_CSS}</style>',
        '</head><body>',
        '<main>',
        '<header>',
        '<div class="title">Chat Index</div>',
        '<div class="subtitle">Full initial queries shown • Click a title to open the Markdown</div>',
        '</header>',
        '<div class="toolbar">',
        '<input id="q" type="search" placeholder="Filter by title, keywords, or date (press / to focus)" aria-label="Filter">',
        '<span class="badge" id="count"></span>',
        '</div>',
        '<div class="pager" id="pager-top">',
        '<button class="pg prev" disabled>Prev</button>',
        '<span class="pages"></span>',
        '<button class="pg next">Next</button>',
        '<select id="pp" class="pp" aria-label="Rows per page"><option>25</option><option selected>50</option><option>100</option></select>',
        '</div>',
        '<div class="table-wrap">',
        '<table>',
        '<thead><tr><th data-sort="link">Link</th><th data-sort="date">Date</th><th data-sort="text">Title</th><th data-sort="text">Keywords</th><th data-sort="num">Messages</th><th data-sort="num">Cluster</th></tr></thead>',
        '<tbody>'
    ]
    for (bn, fn, cnt, html_fn), lab, (r_bn, r_msgs, _) in zip(index, labels, records):
        if ' - ' in bn:
            date, title = bn.split(' - ', 1)
        else:
            date, title = '', bn
        kws = extract_keywords_for_messages(r_msgs)
        full_title = title  # full initial query part
        safe_title = html_escape(full_title)
        # Prefer linking to cleaned HTML if it was generated; otherwise, link to Markdown
        target = html_fn if html_fn else fn
        link = f"<a href='{target}'>{safe_title}</a>"
        copy_btn = (f"<button class='copy' data-href='{target}' title='Copy link to clipboard'>🔗</button>")
        html.append(
            f"<tr>"
            f"<td data-col='link'>{copy_btn}</td>"
            f"<td data-col='date'>{html_escape(date)}</td>"
            f"<td data-col='title'>{link}</td>"
            f"<td data-col='kws'>{html_escape(', '.join(kws))}</td>" # type: ignore
            f"<td data-col='msgs'><span class='badge'>{cnt}</span></td>"
            f"<td data-col='cluster'><span class='badge'>{lab}</span></td>"
            f"</tr>"
        )
    html.extend([
        '</tbody></table></div>',
        '<div class="pager" id="pager-bot">',
        '<button class="pg prev" disabled>Prev</button>',
        '<span class="pages"></span>',
        '<button class="pg next">Next</button>',
        '</div>',
        '<script>',
        '(function(){',
        '  var q = document.getElementById("q");',
        '  var count = document.getElementById("count");',
        '  var table = document.querySelector("table");',
        '  var thead = table && table.querySelector("thead");',
        '  var tbody = table && table.querySelector("tbody");',
        '  if (!tbody) return;',
        '  var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));',
        '  var sortState = {col:null, dir:1}; // 1 asc, -1 desc',
        '  var ppSel = document.getElementById("pp");',
        '  var perPage = parseInt((ppSel && ppSel.value) || "50", 10);',
        '  var current = 0;',
        '  var visible = rows.slice();',
        '  var pagers = Array.prototype.slice.call(document.querySelectorAll(".pager"));',
        '',
        '  function unmark(el){',
        '    if (!el) return;',
        '    el.querySelectorAll("mark").forEach(function(m){',
        '      var t = document.createTextNode(m.textContent);',
        '      m.parentNode.replaceChild(t, m);',
        '      m.parentNode.normalize();',
        '    });',
        '  }',
        '  function highlight(td, term){',
        '    if (!term || !td) return;',
        '    var text = td.textContent;',
        '    var lower = text.toLowerCase();',
        '    var idx = 0; var out = []; var last = 0; var t = term.toLowerCase();',
        '    while ((idx = lower.indexOf(t, idx)) !== -1){',
        '      out.push(text.slice(last, idx));',
        '      out.push("<mark>" + text.slice(idx, idx + t.length) + "</mark>");',
        '      idx += t.length; last = idx;',
        '    }',
        '    out.push(text.slice(last));',
        '    td.innerHTML = out.join("");',
        '  }',
        '  function parseDate(s){',
        '    if (/^\\d{4}-\\d{2}-\\d{2}$/.test(s)) return new Date(s+"T00:00:00Z").getTime();',
        '    var t = Date.parse(s); return isNaN(t) ? 0 : t;',
        '  }',
        '  function sortBy(col, type){',
        '    var dir = (sortState.col === col ? -sortState.dir : 1);',
        '    sortState = {col:col, dir:dir};',
        '    rows.sort(function(a,b){',
        '      var A = a.querySelector("[data-col="+col+"]");',
        '      var B = b.querySelector("[data-col="+col+"]");',
        '      var av = A ? A.textContent.trim() : "";',
        '      var bv = B ? B.textContent.trim() : "";',
        '      if (type === "num"){',
        '        av = parseFloat(av.replace(/[^0-9.-]/g, "")) || 0;',
        '        bv = parseFloat(bv.replace(/[^0-9.-]/g, "")) || 0;',
        '      } else if (type === "date"){',
        '        av = parseDate(av);',
        '        bv = parseDate(bv);',
        '      } else {',
        '        av = av.toLowerCase(); bv = bv.toLowerCase();',
        '      }',
        '      if (av < bv) return -1*dir; if (av > bv) return 1*dir; return 0;',
        '    });',
        '    rows.forEach(function(tr){ tbody.appendChild(tr); });',
        '    // Recalculate visible order to match the new DOM order',
        '    visible = rows.filter(function(tr){ return tr.style.display !== "none"; });',
        '    current = 0;',
        '    renderPage();',
        '    updatePager();',
        '  }',
        '  function applyFilter(){',
        '    var term = (q && q.value || "").trim().toLowerCase();',
        '    visible = [];',
        '    rows.forEach(function(tr){',
        '      ["title","kws","date"].forEach(function(c){ var cell = tr.querySelector("[data-col="+c+"]"); if (cell) unmark(cell); });',
        '      var txt = tr.textContent.toLowerCase();',
        '      var ok = !term || txt.indexOf(term) !== -1;',
        '      tr.__match = ok;',
        '      if (ok){',
        '        visible.push(tr);',
        '        if (term){',
        '          highlight(tr.querySelector("[data-col=title]"), term);',
        '          highlight(tr.querySelector("[data-col=kws]"), term);',
        '        }',
        '      }',
        '    });',
        '    current = 0;',
        '    renderPage();',
        '    updatePager();',
        '  }',
        '  function clamp(n, lo, hi){ return Math.max(lo, Math.min(hi, n)); }',
        '  function totalPages(){ return Math.max(1, Math.ceil(visible.length / perPage)); }',
        '  function renderPage(){',
        '    var start = current * perPage;',
        '    var end = start + perPage;',
        '    rows.forEach(function(tr){ tr.style.display = "none"; });',
        '    visible.forEach(function(tr, i){',
        '      if (i >= start && i < end){ tr.style.display = ""; } else { tr.style.display = "none"; }',
        '    });',
        '    if (count) count.textContent = visible.length + " / " + rows.length;',
        '  }',
        '  function pageButton(num, active){',
        '    return "<button class=\\"pg\\" data-page=\\""+num+"\\" "+(active?"disabled":"")+">"+(num+1)+"</button>";',
        '  }',
        '  function buildPages(){',
        '    var tp = totalPages();',
        '    var start = clamp(current-2, 0, Math.max(0, tp-5));',
        '    var end = Math.min(tp, start+5);',
        '    var html = [];',
        '    if (start > 0){ html.push(pageButton(0, false)); if (start > 1) html.push("<span class=\\"badge\\">…</span>"); }',
        '    for (var i=start;i<end;i++){ html.push(pageButton(i, i===current)); }',
        '    if (end < tp){ if (end < tp-1) html.push("<span class=\\"badge\\">…</span>"); html.push(pageButton(tp-1, false)); }',
        '    return html.join("");',
        '  }',
        '  function updatePager(){',
        '    var tp = totalPages();',
        '    pagers.forEach(function(pg){',
        '      var prev = pg.querySelector(".prev");',
        '      var next = pg.querySelector(".next");',
        '      var pages = pg.querySelector(".pages");',
        '      if (prev) prev.disabled = (current <= 0);',
        '      if (next) next.disabled = (current >= tp-1);',
        '      if (pages) pages.innerHTML = buildPages();',
        '    });',
        '  }',
        '  document.addEventListener("click", function(e){',
        '    var btn = e.target.closest(".pg"); if (!btn) return;',
        '    var trgt = btn.getAttribute("data-page");',
        '    if (trgt != null){',
        '      current = clamp(parseInt(trgt,10)||0, 0, totalPages()-1);',
        '      renderPage(); updatePager(); return;',
        '    }',
        '    if (btn.classList.contains("prev")){ current = clamp(current-1,0,totalPages()-1); renderPage(); updatePager(); return; }',
        '    if (btn.classList.contains("next")){ current = clamp(current+1,0,totalPages()-1); renderPage(); updatePager(); return; }',
        '  }, true);',
        '  if (ppSel){ ppSel.addEventListener("change", function(){ perPage = parseInt(ppSel.value,10)||50; current = 0; renderPage(); updatePager(); }); }',
        '  if (thead){',
        '    thead.addEventListener("click", function(e){',
        '      var th = e.target.closest("th"); if (!th) return;',
        '      var type = th.getAttribute("data-sort"); if (!type) return;',
        '      var idxMap = {"link":"link","date":"date","text":"title","num":"msgs"};',
        '      var col = idxMap[type] || "title";',
        '      // Perform sort first (this will flip sortState.dir if same col)',
        '      sortBy(col, type);',
        '      // Clear previous indicators',
        '      Array.prototype.forEach.call(thead.querySelectorAll(\'th[data-sort]\'), function(h){',
        '        h.classList.remove(\'sorted-asc\',\'sorted-desc\',\'active\');',
        '      });',
        '      // Set indicator on the clicked header based on current sort direction',
        '      th.classList.add(\'active\');',
        '      if (sortState.dir === 1){ th.classList.add(\'sorted-asc\'); }',
        '      else { th.classList.add(\'sorted-desc\'); }',
        '    });',
        '  }',
        '  // Copy link buttons (already handled by delegation above for .pg, do separate for .copy) ',
        '  tbody.addEventListener("click", function(e){',
        '    var btn = e.target.closest(".copy"); if (!btn) return;',
        '    var href = btn.getAttribute("data-href"); if (!href) return;',
        '    btn.setAttribute("aria-busy","true");',
        '    navigator.clipboard.writeText(href).then(function(){',
        '      btn.textContent = "✓"; setTimeout(function(){ btn.textContent = "🔗"; btn.removeAttribute("aria-busy"); }, 900);',
        '    }).catch(function(){',
        '      btn.textContent = "⚠"; setTimeout(function(){ btn.textContent = "🔗"; btn.removeAttribute("aria-busy"); }, 1200);',
        '    });',
        '  });',
        '  // Sticky header shadow on scroll',
        '  var wrap = document.querySelector(".table-wrap");',
        '  window.addEventListener("scroll", function(){',
        '    if (!wrap) return;',
        '    var y = wrap.getBoundingClientRect().top;',
        '    document.body.classList.toggle("thead-shadow", y < 0);',
        '  });',
        '  // Init',
        '  applyFilter();',
        '})();',
        '</script>',
        '</main>',
        '</body></html>'
    ])

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
