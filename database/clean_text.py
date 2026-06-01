# database/clean_text.py
import re


def clean_text(text):
    if not text:
        return ""

    text = text.lower()

    text = text.replace("\n", " ")

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

