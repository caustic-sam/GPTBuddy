import os
import json
import re
from pathlib import Path
from datetime import datetime

# Define the input and output paths
INPUT_FILE = "./raw_chats/openai_export.json"   # <-- replace this with your actual filename
OUTPUT_DIR = "./parsed_chats/"         # Folder to store the markdown output

def sanitize_filename(title):
    """
    Converts a chat title into a safe filename by removing illegal characters.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", title)

def parse_chat_file(input_path):
    """
    Parses the raw OpenAI export JSON and returns a list of (title, markdown_text) tuples.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    parsed_chats = []

    for convo in raw_data:
        title = convo.get("title", "Untitled Conversation")
        mapping = convo.get("mapping", {})
        messages = []

        # Extract all message entries
        for node_id, node_data in mapping.items():
            msg = node_data.get("message")
            if not msg:
                continue

            role = msg["author"]["role"]
            content = "\n".join(msg["content"]["parts"])
            timestamp = datetime.fromtimestamp(msg.get("create_time", 0))

            messages.append((timestamp, role, content))

        # Sort messages by timestamp to maintain conversation order
        messages.sort(key=lambda x: x[0])

        # Build markdown content for this conversation
        markdown = f"# {title}\n\n"
        for timestamp, role, content in messages:
            role_label = "**You:**" if role == "user" else "**Assistant:**"
            markdown += f"### {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            markdown += f"{role_label}\n\n{content.strip()}\n\n"

        parsed_chats.append((title, markdown))

    return parsed_chats

def save_markdown_files(chats, output_dir):
    """
    Saves each parsed chat as a separate .md file in the output directory.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for title, content in chats:
        filename = sanitize_filename(title) + ".md"
        output_path = Path(output_dir) / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ Saved: {output_path}")

# Run the full pipeline
if __name__ == "__main__":
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_DIR)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path.resolve()}")
    else:
        print(f"📥 Parsing: {input_path.name}")
        chats = parse_chat_file(input_path)
        save_markdown_files(chats, output_path)
        print(f"\n📚 Done! All parsed chats are saved in: {output_path.resolve()}")
