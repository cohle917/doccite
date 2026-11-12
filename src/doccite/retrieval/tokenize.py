import re
import unicodedata
from functools import lru_cache

import jieba
from nltk.stem.snowball import EnglishStemmer

_WORDS = re.compile(r"[\u3400-\u9fff]+|\d+(?:[.,]\d+)*|[^\W\d_]+", re.UNICODE)
_STEM = EnglishStemmer().stem
_STOP = frozenset(
    "a an the this that of on in at to and or for by is are was were be been with from "
    "what when where who how which does did do 的 了 吗 呢 是 和 在 与 请 什么 多少".split()
)


@lru_cache(maxsize=4096)
def tokenize(text: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", text).casefold()
    tokens = []
    for match in _WORDS.finditer(text):
        word = match[0]
        if "\u3400" <= word[0] <= "\u9fff":
            tokens.extend(term for term in jieba.lcut(word) if term not in _STOP)
        elif word not in _STOP:
            tokens.append(_STEM(word) if word.isascii() and word.isalpha() else word)
    return tuple(tokens)
