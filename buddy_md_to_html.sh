#!/usr/bin/env bash
set -euo pipefail

# Convert each Markdown file to HTML using Pandoc
for md in *.md; do
  html="${md%.md}.html"
  echo "Converting '$md' → '$html'"
  pandoc --standalone --css=style.css "$md" -o "$html"
done

echo "Updating index page links to .html"
sed -i '' 's/\.md">/\.html">/g' chat_index.html
sed -i '' "s/\.md'>/\.html'>/g" chat_index.html