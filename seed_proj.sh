bash -lc 'set -euo pipefail;
# --- repo + milestone setup ---
REPO_FULL="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"; OWNER="${REPO_FULL%%/*}"; REPO="${REPO_FULL#*/}";
MS="v0.1.0";
# Ensure milestone exists
if ! gh api "repos/$OWNER/$REPO/milestones" --jq ".[].title" | grep -Fxq "$MS"; then
  gh api "repos/$OWNER/$REPO/milestones" -f title="$MS" -f state=open -f description="First public cut with packaging, tests, and HTML index" >/dev/null
  echo "Created milestone $MS"
else
  echo "Milestone $MS already exists"
fi
# --- labels (idempotent) ---
for args in \
  "type:enhancement 3E4B9E New feature" \
  "type:bug D73A4A Defect" \
  "type:docs 0E8A16 Docs/Readme" \
  "type:ci 5319E7 CI/CD" \
  "priority:high B60205 High priority" \
  "good first issue 7057ff Onboarding friendly"
do
  name="${args%% *}"; rest="${args#* }"; color="${rest%% *}"; desc="${rest#* }";
  gh label create "$name" --color "$color" --description "$desc" 2>/dev/null || true
done
# --- helper to create + milestone an issue ---
mk(){ local T="$1"; local B="$2"; local L="$3";
  NUM="$(gh issue create -t "$T" -b "$B" -l "$L" --json number --jq .number)"
  gh issue edit "$NUM" --milestone "$MS" >/dev/null
  echo "• opened #$NUM  $T"
}
# --- seed issues ---
mk "Packaging: pyproject.toml + console script (gptbuddy)" "Package as installable CLI. Add entry point to run the organizer. Include version + classifiers." "type:enhancement,priority:high"
mk "Output layout: md/ and html/ + fix all links" "Per‑chat Markdown → output_dir/md/; per‑chat HTML → output_dir/html/. Keep index files at root. Update links in index.{md,html} and clusters_index.md." "type:enhancement,priority:high"
mk "README: quickstart + screenshots/GIF + large export notes" "Document CLI usage, examples, and performance tips. Add a GIF of index.html filtering/pagination." "type:docs"
mk "Unit tests: mapping extraction & cluster guard rails" "Golden tests for mapping extraction; ensure fallback when docs are empty/stopword‑only." "type:enhancement"
mk "Unit tests: HTML cleaner patterns" "Cover removal of textdoc_id/jawbone_id artifacts and fenced JSON blocks. Add samples." "type:enhancement"
mk "CI: lint + tests + build wheel (local only)" "GitHub Actions: ruff/flake8, pytest, cache pip, build wheel as artifact (no PyPI publish)." "type:ci"
mk "Deploy scripts: local release helper" "Scripts to build wheel, tag, and draft a GitHub Release with artifacts." "type:enhancement"
mk "WordPress‑friendly HTML export" "Minimal inline CSS, strong sanitization, light/dark compatibility; validate copy‑link buttons." "type:enhancement"
mk "Docs: example dataset + demo screencap" "Ship a tiny mock export for screenshots and a short clip of the index UI." "type:docs"
echo "✅ Done seeding milestone, labels, and issues in $REPO_FULL"
'
