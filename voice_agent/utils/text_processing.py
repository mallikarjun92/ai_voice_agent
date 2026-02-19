import re

def sanitize_text(text):
    """Removes emojis, markdown, and special characters for clean TTS."""
    if not text:
        return ""
    # Remove Emojis and non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove Markdown symbols like *, _, #
    text = re.sub(r'[*_#`~\[\]()]', '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences(text):
    """Splits text into sentences based on punctuation."""
    # Split at . ! or ? followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
