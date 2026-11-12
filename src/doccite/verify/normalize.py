import unicodedata

_PUNCTUATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "「": '"',
        "」": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        "。": ".",
        "、": ",",
    }
)


def normalize(text: str) -> tuple[str, list[tuple[int, int]]]:
    characters = []
    offsets = []
    start = 0
    while start < len(text):
        end = start + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        token = unicodedata.normalize("NFKC", text[start:end]).translate(_PUNCTUATION)
        for char in token:
            if char.isspace():
                if characters and characters[-1] == " ":
                    offsets[-1] = (offsets[-1][0], end)
                elif characters:
                    characters.append(" ")
                    offsets.append((start, end))
            else:
                characters.append(char)
                offsets.append((start, end))
        start = end
    if characters and characters[-1] == " ":
        characters.pop()
        offsets.pop()
    return "".join(characters), offsets
