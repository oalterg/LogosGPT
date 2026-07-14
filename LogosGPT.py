"""
LogosGPT — word pieces, a wider canon, a web UI, and a KV cache.

The corpus is a canon of scripture and Western wisdom in elevated English,
chosen around the theme of the Logos (the Word / ordering reason) and Truth:
the King James Bible and its Wisdom-of-Solomon/Sirach Apocrypha, Heraclitus
(the first speaker of the Logos), Plato (thirteen dialogues, from the Republic
to the Cratylus on whether names tell the truth), Aristotle (Nicomachean
Ethics), Cicero, Lucretius, Marcus Aurelius, Epictetus (Enchiridion and
Discourses), Seneca, Philo of Alexandria (who first gave the Logos a person),
Plotinus (the Enneads), Iamblichus, the Golden Verses of Pythagoras, Justin
Martyr and Clement of Alexandria (the Word sown among the Greeks), Athanasius
(the Word made flesh), Augustine, Boethius, Aquinas (Summa Theologica, Prima
Pars), Dante (the Divine Comedy), Thomas a Kempis, Pascal, Bunyan, Milton,
Kierkegaard, the Corpus Hermeticum (the Divine Pymander), and the Kybalion.

The tokenizer is byte-pair encoding (BPE) trained on this corpus: it starts
from single characters and greedily merges the most frequent adjacent pair
until the vocabulary is full. Common words become one token, rare names are
spelled from pieces — so nothing is ever out of vocabulary, unlike the
word-level tokenizer this replaces. Tokens carry their leading space
(" the"), so decoding is pure concatenation.

The transformer is GPT-2-style (rmsnorm, no biases, squared ReLU, rotary
positions, tied embeddings) with one addition: a reserved "source" token per book, e.g.
<meditations>. It leads every training window (window-level tagging, so the
model sees it constantly rather than once per book), so generation can be
steered to one voice — Marcus Aurelius, or Milton, or the KJV. On a fraction of
windows the source token is instead <canon> (conditioning dropout), which trains
a joint voice over the whole corpus at once — the "consider everything" mode you
get when no single source is chosen. A last reserved token, <end>, marks each
passage boundary in training (the corpus is one passage per line), so generation
stops when the model emits it instead of running to a fixed length. Generation
uses a KV cache: each new token attends to the stored keys/values of its
predecessors, not the whole prefix.

Provenance and retrieval: source tags make generation attributable to a voice,
but they do not make it true — every generated verse is still newly invented.
Truthfulness is retrieval's job: a BM25 index over the sources returns real
passages verbatim, each with a citation you can check (e.g. "Genesis 1:1").
Ask for wisdom and the scribe writes; ask where it is written and the index
points you to the source, unedited.

Usage:
    python LogosGPT.py                # train, then sample; saves logosgpt.pt
    python LogosGPT.py --steps 200    # quick smoke test
    python LogosGPT.py sample         # generate from logosgpt.pt, no training
    python LogosGPT.py chat           # interactive completion (':<voice>' to pick a voice)
    python LogosGPT.py search "mercy" # cite real passages from the canon (no model needed)
    python LogosGPT.py web            # serve the web UI on http://127.0.0.1:8000
"""

import heapq
import html
import json
import math
import os
import re
import sys
import textwrap
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)  # Let there be order among chaos

# Model / training configuration — sized for ~48GB of unified memory (M-series).
# On an 8GB machine, drop batch_size to 8 and expect ~1s/step.
n_layer = 8        # depth of the transformer
n_embd = 384       # width of the network (embedding dimension)
n_head = 6         # number of attention heads (head dim 64)
block_size = 512   # context length in tokens (~1,800 characters of text)
batch_size = 128   # sequences per training step
vocab_target = 4096  # BPE vocabulary size: base characters + learned merges
num_steps = 5000   # ~4h on an M5 Pro; the best-val checkpoint is kept, so interrupting is safe
learning_rate = 1e-3
warmup_steps = 100    # linear ramp from 0, so early steps don't jolt the fresh weights
min_lr = 1e-4         # floor of the cosine decay (~lr/10): keep learning to the end
dropout = 0.1      # regularization: the corpus is small and the epochs are many
weight_decay = 0.1    # on matrices only, not rmsnorm gains
blend_dropout = 0.15  # fraction of windows tagged <canon> (the joint voice) not their source
eval_every = 250   # how often to measure validation loss (and save, if it improved)

num_steps = int(sys.argv[sys.argv.index('--steps') + 1]) if '--steps' in sys.argv else num_steps
chat_mode = 'chat' in sys.argv
web_mode = 'web' in sys.argv
search_mode = 'search' in sys.argv  # pure retrieval; needs the corpus but not the model
sample_only = 'sample' in sys.argv or chat_mode or web_mode

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

# The canon: one Project Gutenberg text per book, cleaned to one paragraph per line
BOOKS = {
    'bible.txt': 'https://www.gutenberg.org/cache/epub/10/pg10.txt',
    'meditations.txt': 'https://www.gutenberg.org/cache/epub/2680/pg2680.txt',
    'enchiridion.txt': 'https://www.gutenberg.org/cache/epub/45109/pg45109.txt',
    'imitation_of_christ.txt': 'https://www.gutenberg.org/cache/epub/1653/pg1653.txt',
    'confessions_augustine.txt': 'https://www.gutenberg.org/cache/epub/3296/pg3296.txt',
    'the_republic.txt': 'https://www.gutenberg.org/cache/epub/1497/pg1497.txt',
    'pilgrims_progress.txt': 'https://www.gutenberg.org/cache/epub/131/pg131.txt',
    'paradise_lost.txt': 'https://www.gutenberg.org/cache/epub/26/pg26.txt',
    # Western wisdom & philosophy on the Logos / Truth theme
    'timaeus.txt': 'https://www.gutenberg.org/cache/epub/1572/pg1572.txt',            # Plato, the cosmos ordered by reason
    'boethius.txt': 'https://www.gutenberg.org/cache/epub/14328/pg14328.txt',         # Consolation of Philosophy
    'epictetus_discourses.txt': 'https://www.gutenberg.org/cache/epub/10661/pg10661.txt',  # the fuller Stoic teaching
    'pascal_pensees.txt': 'https://www.gutenberg.org/cache/epub/18269/pg18269.txt',   # reason, truth, and its limits
    'seneca_morals.txt': 'https://www.gutenberg.org/cache/epub/56075/pg56075.txt',    # Stoic morals (L'Estrange, 1678)
    # KJV Apocrypha, sliced to the two wisdom books (see slice_apocrypha)
    'apocrypha.txt': 'https://www.gutenberg.org/cache/epub/124/pg124.txt',            # Wisdom of Solomon + Sirach
    # The Corpus Hermeticum is not on Gutenberg; assembled from Wikisource instead
    'hermetica.txt': 'https://en.wikisource.org/wiki/The_Divine_Pymander',            # Poemandres + Books 2-3
    'kybalion.txt': 'https://www.gutenberg.org/cache/epub/14209/pg14209.txt',         # Hermetic philosophy (1908)
    # Reason and the Logos: more Plato, then Aristotle, Aquinas, Kierkegaard
    'apology.txt': 'https://www.gutenberg.org/cache/epub/1656/pg1656.txt',             # Socrates on trial for the truth
    'phaedo.txt': 'https://www.gutenberg.org/cache/epub/1658/pg1658.txt',              # the soul, and philosophy as practice for death
    'symposium.txt': 'https://www.gutenberg.org/cache/epub/1600/pg1600.txt',           # the ascent to Beauty itself
    'nicomachean_ethics.txt': 'https://www.gutenberg.org/cache/epub/8438/pg8438.txt',  # Aristotle on virtue and practical wisdom
    'summa_theologica.txt': 'https://www.gutenberg.org/cache/epub/17611/pg17611.txt',  # Aquinas, Prima Pars: God known by reason
    'kierkegaard.txt': 'https://www.gutenberg.org/cache/epub/60333/pg60333.txt',       # truth as inward earnestness (Selections)
    # The wider ancient Logos: Heraclitus (its first speaker) to the Neoplatonists
    'heraclitus.txt': 'https://en.wikisource.org/wiki/Fragments_of_Heraclitus_(annotated)',  # Burnet's fragments, via Wikisource
    'plotinus.txt': 'https://www.gutenberg.org/cache/epub/42930/pg42930.txt',          # Enneads: Guthrie's 4 volumes (see fetch_plotinus)
    'cicero.txt': 'https://www.gutenberg.org/cache/epub/14988/pg14988.txt',            # Tusculan Disputations, Nature of the Gods, Commonwealth
    'iamblichus.txt': 'https://www.gutenberg.org/cache/epub/72815/pg72815.txt',        # On the Mysteries (Thomas Taylor)
    'golden_verses.txt': 'https://www.gutenberg.org/cache/epub/69174/pg69174.txt',     # the Verses + d'Olivet's Examinations (sliced)
    'lucretius.txt': 'https://www.gutenberg.org/cache/epub/785/pg785.txt',             # On the Nature of Things (Leonard's verse)
    # The Logos itself, traced from Athens to Alexandria to John: Philo gives the
    # Word a person, and the Alexandrians receive it. Only Clement is on Gutenberg.
    'philo.txt': 'http://www.earlychristianwritings.com/yonge/',                       # the allegorical commentary (see fetch_philo)
    'justin_martyr.txt': 'https://en.wikisource.org/wiki/Writings_of_Justin_Martyr',   # both Apologies + Trypho (see fetch_justin)
    'athanasius.txt': 'https://en.wikisource.org/wiki/Nicene_and_Post-Nicene_Fathers:_Series_II',  # On the Incarnation (see fetch_athanasius)
    'clement_alexandria.txt': 'https://www.gutenberg.org/cache/epub/71937/pg71937.txt',  # Exhortation, Instructor, Stromata (see fetch_clement)
    # More Plato, chosen where the dialogues turn on the word and the truth
    'cratylus.txt': 'https://www.gutenberg.org/cache/epub/1616/pg1616.txt',            # whether names are true by nature or convention
    'phaedrus.txt': 'https://www.gutenberg.org/cache/epub/1636/pg1636.txt',            # the soul's chariot; the living word against the written
    'theaetetus.txt': 'https://www.gutenberg.org/cache/epub/1726/pg1726.txt',          # what knowledge is, and is not
    'sophist.txt': 'https://www.gutenberg.org/cache/epub/1735/pg1735.txt',             # being, not-being, and how speech can be false
    'parmenides.txt': 'https://www.gutenberg.org/cache/epub/1687/pg1687.txt',          # the One: the seed of all later Neoplatonism
    'gorgias.txt': 'https://www.gutenberg.org/cache/epub/1672/pg1672.txt',             # rhetoric against truth
    'meno.txt': 'https://www.gutenberg.org/cache/epub/1643/pg1643.txt',                # virtue, and knowledge as recollection
    'philebus.txt': 'https://www.gutenberg.org/cache/epub/1744/pg1744.txt',            # mind, pleasure, and the reason ordering the cosmos
    # The Logos as a poem: the cosmos as "the love that moves the sun and other stars"
    'dante.txt': 'https://www.gutenberg.org/cache/epub/1004/pg1004.txt',               # the Divine Comedy, Longfellow's blank verse
}

# Some sources need a slice applied after the Gutenberg header/footer is stripped
# and before paragraphs are unwrapped: fetch_book consults this by filename.
def slice_apocrypha(text):
    """Keep only the two wisdom books of the KJV Apocrypha: Wisdom of Solomon and
    Sirach (Ecclesiasticus), dropping Tobit, Judith, Maccabees and the rest."""
    start = text.find('[The Wisdom of Solomon]')
    end = text.find('The Book of Baruch', start)  # the (unbracketed) heading after Sirach
    return text[start:end] if start != -1 and end != -1 else text

def slice_golden_verses(text):
    """Keep the Golden Verses themselves and d'Olivet's esoteric Examinations,
    dropping his long opening discourse on the essence and form of poetry."""
    start = text.find('THE GOLDEN VERSES OF PYTHAGORAS')  # the first all-caps heading
    return text[start:] if start != -1 else text

SLICERS = {'apocrypha.txt': slice_apocrypha, 'golden_verses.txt': slice_golden_verses}

def strip_gutenberg(raw):
    """Normalize a Project Gutenberg download and cut its header and footer."""
    raw = raw.replace('\r', '').lstrip('﻿')
    start, end = raw.find('*** START'), raw.find('*** END')
    if start != -1 and end != -1:
        raw = raw[raw.index('\n', start) + 1:end]
    return raw

def fetch_hermetica():
    """Assemble the Divine Pymander (Hermes Trismegistus, Everard's 1650 English)
    from Wikisource's clean, hand-transcribed pages — the Corpus Hermeticum has no
    Project Gutenberg edition. Only Books 1-3 are transcribed there, but Book 1 is
    the Poemandres, the Hermetic Logos text itself."""
    ua = {'User-Agent': 'DuckGPT-corpus/1.0 (educational research)'}
    def api(**params):
        params.update(format='json', formatversion='2')
        u = 'https://en.wikisource.org/w/api.php?' + urllib.parse.urlencode(params)
        return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=ua), timeout=60).read().decode('utf-8', 'replace'))
    listing = api(action='query', list='allpages', apnamespace='0',
                  apprefix='The Divine Pymander/Book ', aplimit='50')['query']['allpages']
    pages = sorted((p['title'] for p in listing),
                   key=lambda t: int(re.search(r'(\d+)', t.rsplit('/', 1)[-1]).group(1)))
    books = []
    for title in pages:
        h = api(action='parse', page=title, prop='text')['parse']['text']
        h = re.sub(r'(?is)<(style|table).*?</\1>', ' ', h)
        h = re.sub(r'(?is)<sup[^>]*>.*?</sup>', ' ', h)  # footnote and page-number marks
        text = html.unescape(re.sub(r'(?s)<[^>]+>', ' ', h))
        text = re.split(r'Layout \d+', text, maxsplit=1)[-1]  # drop the Wikisource page header
        text = text.replace('​', '').replace('﻿', '')  # zero-width marks between elements
        text = re.sub(r'(?m)^\s*\d+\.\s+', '', text)          # verse numbers at line start
        text = re.sub(r'(?<=[.?!;:])\s+\d+\.\s+', ' ', text)  # verse numbers after a verse ends
        books.append(text)
    return '\n\n'.join(books)

def fetch_heraclitus():
    """The fragments of Heraclitus — the first Logos text of all — in Burnet's
    1920 translation, from Wikisource (no Gutenberg edition). One fragment per
    line, with Bywater numbers, footnote marks, and R.P. citations removed."""
    ua = {'User-Agent': 'DuckGPT-corpus/1.0 (educational research)'}
    u = ('https://en.wikisource.org/w/api.php?action=parse&prop=text&format=json'
         '&formatversion=2&page=' + urllib.parse.quote('Fragments of Heraclitus (annotated)'))
    h = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=ua), timeout=60)
                   .read().decode('utf-8', 'replace'))['parse']['text']
    h = re.sub(r'(?is)<(style|table).*?</\1>', ' ', h)
    h = re.sub(r'(?is)<sup[^>]*>.*?</sup>', ' ', h)  # footnote marks
    text = html.unescape(re.sub(r'(?s)<[^>]+>', ' ', h)).replace('​', '')
    frags = re.split(r'Fragment\s+[\d\-]+a?\s*\[\s*edit\s*\]', text)[1:]  # [0] is header + contents
    hints = {'the', 'of', 'and', 'to', 'is', 'in', 'that', 'it', 'not', 'are',
             'a', 'an', 'for', 'we', 'men', 'all', 'as', 'be'}
    out = []
    for f in frags:
        f = re.split(r'Footnotes\s*\[\s*edit\s*\]|↑', f)[0]  # the last fragment drags the endnotes
        f = re.sub(r'\[([A-Za-z ]{1,20})\]', r'\1', f)       # unbracket editorial glosses, e.g. [is]
        f = re.sub(r'\[[^\]]*\]', '', f)                     # drop source attributions [Hisdosus ...]
        f = re.sub(r'\(\d+[^)]{0,8}\)', '', f)               # Bywater numbers, e.g. (126), (105-107)
        f = re.sub(r'R\s?\.\s?P\s?\.\s?\d+\s?[a-eA-E]?\s?\.?', '', f)  # Ritter-Preller citations
        f = ' '.join(f.split())
        words = f.lower().split()
        # keep what reads as Burnet's English; drop untranslated Greek/Latin apparatus
        if words and sum(w in hints for w in words) >= 0.1 * len(words):
            out.append(f)
    return '\n\n'.join(out)  # blank lines survive fetch_book's paragraph unwrap

def fetch_plotinus():
    """Plotinus' complete works (the Enneads, with Porphyry's biography), Guthrie's
    1918 translation: four Project Gutenberg volumes joined into one source."""
    volumes = []
    for vid in (42930, 42931, 42932, 42933):
        u = f'https://www.gutenberg.org/cache/epub/{vid}/pg{vid}.txt'
        print(f"  + {u}")
        volumes.append(strip_gutenberg(urllib.request.urlopen(u).read().decode('utf-8')))
    return '\n\n'.join(volumes)

def fetch_philo():
    """Philo of Alexandria's allegorical commentary on Genesis, in Yonge's 1854
    translation. Philo is the hinge of this canon: the Alexandrian Jew who read Moses
    with Plato's eyes and gave the Logos a person — God's firstborn, his image, the
    instrument by which the world was made — a lifetime before John's prologue opened
    with the same word. Kept here are the twenty-one treatises of the commentary
    proper (On the Creation through On Dreams), where that doctrine is worked out;
    his legal, historical and apologetic books are left aside. There is no Gutenberg
    edition, so the text comes from earlychristianwritings.com, one treatise a page."""
    ua = {'User-Agent': 'DuckGPT-corpus/1.0 (educational research)'}
    books = []
    for n in range(1, 22):  # book1 (On the Creation) .. book21 (On Dreams)
        u = f'http://www.earlychristianwritings.com/yonge/book{n}.html'
        print(f"  + {u}")
        h = urllib.request.urlopen(urllib.request.Request(u, headers=ua), timeout=60)
        h = h.read().decode('iso-8859-1')
        h = h[h.find('<div id="textboundingbox">'):h.find('<div id="footer')]
        h = h[:h.rfind('<hr')]  # the rule closing the treatise, above the site's own navigation
        h = re.sub(r'(?s)<!--.*?-->', ' ', h)
        h = re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>', ' ', h)  # ad slots
        h = re.sub(r'(?is)<h1[^>]*>.*?</h1>', ' ', h)              # the site's title banner
        h = re.sub(r'\{[^}]*\}', '', h)  # Yonge's footnotes, scripture refs, editorial asides
        h = re.sub(r'(?i)<p\b[^>]*>', '\n', h)  # paragraphs are opened here, never closed
        text = html.unescape(re.sub(r'(?s)<[^>]+>', '', h))
        books.append('\n\n'.join(' '.join(p.split()) for p in text.split('\n') if p.strip()))
    return '\n\n'.join(books)

def wikisource_text(page):
    """One Wikisource page as text, from its rendered HTML: for pages transcluded from
    proofread scans, where the wikitext is only a transclusion directive."""
    ua = {'User-Agent': 'DuckGPT-corpus/1.0 (educational research)'}
    u = ('https://en.wikisource.org/w/api.php?action=parse&prop=text&format=json'
         '&formatversion=2&page=' + urllib.parse.quote(page))
    h = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=ua), timeout=60)
                   .read().decode('utf-8', 'replace'))['parse']['text']
    i = h.find('id="ws-data"')  # the hidden id/title/author block closes the page header
    if i != -1:
        h = h[h.find('</div>', h.find('</div>', i) + 6) + 6:]
    h = re.split(r'<div[^>]*class="[^"]*licenseContainer', h)[0]  # the licence banner below
    h = re.split(r'<h\d[^>]*id="Footnotes"', h)[0]                # the endnote heading
    h = re.sub(r'(?is)<(style|table|sup)[^>]*>.*?</\1>', ' ', h)  # css, page tables, note marks
    h = re.sub(r'(?is)<ol class="references".*?</ol>', ' ', h)    # the endnotes themselves
    h = h.replace('\n', ' ')  # a line break in the source is not a paragraph break
    h = re.sub(r'(?is)<br\s*/?>|</(p|div|h\d|li|dd|blockquote)>', '\n', h)  # but a block end is
    text = html.unescape(re.sub(r'(?s)<[^>]+>', '', h))
    text = text.replace('​', '').replace('﻿', '')  # zero-width marks between elements
    return '\n\n'.join(' '.join(p.split()) for p in re.sub(r'\[\s*edit\s*\]', ' ', text).split('\n') if p.strip())

def fetch_justin():
    """Justin Martyr's two Apologies and the Dialogue with Trypho (Dods' translation,
    from Wikisource's Ante-Nicene Christian Library). Justin is this canon's argument
    for itself: he holds that the Logos was sown as seed in every people, so that
    whoever lived by reason lived with the Word — and names Socrates and Heraclitus,
    both of them voices here, among those who did."""
    pages = ['Ante-Nicene Christian Library/The First Apology of Justin Martyr',
             'Ante-Nicene Christian Library/The Second Apology of Justin Martyr',
             'Ante-Nicene Christian Library/Dialogue with Trypho']
    out = []
    for p in pages:
        print(f"  + {p}")
        out.append(wikisource_text(p))
    return '\n\n'.join(out)

def fetch_athanasius():
    """Athanasius' On the Incarnation, the classic treatise on the Word made flesh
    (Nicene and Post-Nicene Fathers, Series II, from Wikisource). Its fifty-seven
    chapters are plain wikitext, and the API hands back forty at a time — so this
    takes the wikitext directly rather than asking the parser for page after page."""
    base = ('Nicene and Post-Nicene Fathers: Series II/Volume IV/Incarnation of the Word/'
            'On the Incarnation of the Word/Chapter ')
    titles = [f'{base}{n}' for n in range(1, 58)]
    ua = {'User-Agent': 'DuckGPT-corpus/1.0 (educational research)'}
    pages = {}
    for i in range(0, len(titles), 40):
        print(f"  + On the Incarnation of the Word, chapters {i + 1}-{min(i + 40, len(titles))}")
        params = {'action': 'query', 'prop': 'revisions', 'rvprop': 'content',
                  'rvslots': 'main', 'titles': '|'.join(titles[i:i + 40]),
                  'format': 'json', 'formatversion': '2'}
        u = 'https://en.wikisource.org/w/api.php?' + urllib.parse.urlencode(params)
        d = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=ua), timeout=60).read())
        pages.update({p['title']: p['revisions'][0]['slots']['main']['content']
                      for p in d['query']['pages']})
    chapters = []
    for t in titles:
        wt = pages[t]
        wt = re.sub(r'(?s)\{\{header.*?\n\}\}', '', wt)   # the page's navigation header
        wt = re.split(r'(?m)^==\s*Footnotes\s*==', wt)[0]  # the endnote section
        wt = re.sub(r'(?s)<ref[^>]*>.*?</ref>|<ref[^>]*/>', '', wt)  # Schaff's footnotes
        wt = re.sub(r'(?s)<!--.*?-->', '', wt)
        wt = re.sub(r'\{\{[^|{}]*\|([^{}]*)\}\}', r'\1', wt)  # {{small-caps|Whereas}} -> Whereas
        wt = re.sub(r'\[\[[^\]|]*\|([^\]]*)\]\]', r'\1', wt)  # [[target|shown]] -> shown
        wt = re.sub(r'\[\[([^\]]*)\]\]', r'\1', wt)           # [[shown]] -> shown
        wt = re.sub(r"'{2,}", '', wt)                          # ''italic'' / '''bold'''
        wt = re.sub(r'(?m)^=+\s*(.*?)\s*=+\s*$', r'\1', wt)    # == headings ==
        wt = html.unescape(re.sub(r'(?s)<[^>]+>', '', wt))
        chapters.append('\n\n'.join(' '.join(p.split()) for p in re.split(r'\n\s*\n', wt) if p.strip()))
    return '\n\n'.join(chapters)

def fetch_clement():
    """Clement of Alexandria's Exhortation to the Heathen, the Instructor and the
    Stromata (Ante-Nicene Christian Library): two Project Gutenberg volumes joined.
    Clement is Justin's claim worked out at length — the Logos is the teacher of the
    whole race, and Greek philosophy was the schooling that led the Greeks to him."""
    volumes = []
    for vid in (71937, 73020):
        u = f'https://www.gutenberg.org/cache/epub/{vid}/pg{vid}.txt'
        print(f"  + {u}")
        volumes.append(strip_gutenberg(urllib.request.urlopen(u).read().decode('utf-8')))
    return '\n\n'.join(volumes)

FETCHERS = {'hermetica.txt': fetch_hermetica, 'heraclitus.txt': fetch_heraclitus,
            'plotinus.txt': fetch_plotinus, 'philo.txt': fetch_philo,
            'justin_martyr.txt': fetch_justin, 'athanasius.txt': fetch_athanasius,
            'clement_alexandria.txt': fetch_clement}  # sources built by custom code, not a plain download

# One reserved "source" token per book, e.g. <meditations>. Prepended to every
# training window (see get_batch) so the model learns to condition its voice on
# the source instead of averaging all eight; the stem ('meditations') is the key
# used everywhere — in training, in generation, and over the wire from the UI.
SOURCES = {name: '<' + name[:-4] + '>' for name in BOOKS}  # filename -> token string
SOURCE_TOKENS = list(SOURCES.values())
# One more reserved token for the *joint* voice: the whole canon at once, no single
# source. On a fraction of training windows (see blend_dropout) the source token is
# replaced by <canon>, so the model learns an in-distribution "consider everything"
# mode instead of leaving untagged generation off-distribution.
BLEND_TOKEN = '<canon>'
END_TOKEN = '<end>'  # end of a passage: trained in at line breaks, generation stops on it
CONTROL_TOKENS = SOURCE_TOKENS + [BLEND_TOKEN]  # lead the context, never generated (masked out)
SPECIAL_TOKENS = CONTROL_TOKENS + [END_TOKEN]   # all reserved ids, appended after the BPE vocab

def fetch_book(name, url):
    path = os.path.join('data', name)
    if not os.path.exists(path):
        if name in FETCHERS:  # sources with no plain-text download, assembled by custom code
            print(f"assembling {url} -> {path}")
            raw = FETCHERS[name]()
        else:
            print(f"downloading {url} -> {path}")
            raw = strip_gutenberg(urllib.request.urlopen(url).read().decode('utf-8'))
            if name in SLICERS:  # keep only the wanted span of a larger source
                raw = SLICERS[name](raw)
        raw = re.sub(r'\[\d+\]', '', raw)  # footnote markers, e.g. [267]
        for smart, plain in (('’', "'"), ('‘', "'"), ('“', '"'), ('”', '"')):
            raw = raw.replace(smart, plain)
        # unwrap hard-wrapped paragraphs so each becomes a single line
        paragraphs = [' '.join(p.split()) for p in raw.split('\n\n')]
        open(path, 'w').write('\n'.join(p for p in paragraphs if p) + '\n')
    return open(path).read()

# Pre-tokenizer: split text into "words" that BPE merges may not cross.
# Each word keeps its leading space, so decoding tokens is pure concatenation.
WORD_RE = re.compile(r"\n| ?[A-Za-z']+| ?\d+| ?[^A-Za-z\d\s]+")

def train_bpe(word_freq, target):
    """Learn BPE merges over a {word: count} table until `target` vocab size."""
    t0 = time.time()
    base = sorted(set(ch for w in word_freq for ch in w))
    words = [list(w) for w in word_freq]
    freqs = list(word_freq.values())
    pair_counts = Counter()
    pair_words = defaultdict(set)  # pair -> indices of words that (may) contain it
    for i, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += freqs[i]
            pair_words[pair].add(i)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    while len(base) + len(merges) < target and heap:
        # pop until we find a pair whose heap entry is not stale
        neg_c, best = heapq.heappop(heap)
        if pair_counts.get(best, 0) != -neg_c or neg_c == 0:
            continue
        merges.append(best)
        new_sym = best[0] + best[1]
        touched = set()
        for i in pair_words.pop(best, ()):
            w, f = words[i], freqs[i]
            if not any((w[j], w[j + 1]) == best for j in range(len(w) - 1)):
                continue  # stale index: this word was rewritten by an earlier merge
            for pair in zip(w, w[1:]):  # retract this word's old pair counts
                pair_counts[pair] -= f
                touched.add(pair)
            j, nw = 0, []
            while j < len(w):
                if j < len(w) - 1 and (w[j], w[j + 1]) == best:
                    nw.append(new_sym)
                    j += 2
                else:
                    nw.append(w[j])
                    j += 1
            words[i] = nw
            for pair in zip(nw, nw[1:]):  # add the new pair counts back
                pair_counts[pair] += f
                pair_words[pair].add(i)
                touched.add(pair)
        for pair in touched:
            if pair_counts.get(pair, 0) > 0:
                heapq.heappush(heap, (-pair_counts[pair], pair))
            else:
                pair_counts.pop(pair, None)
        if len(merges) % 1000 == 0:
            print(f"  bpe: {len(merges):5d} merges | last: {best[0]!r}+{best[1]!r} | {time.time()-t0:.0f}s")
    return base, merges

def make_tokenizer(base, merges):
    """Install itos/stoi/encode as globals from a BPE (base, merges) pair.
    The reserved source tokens are appended after the BPE vocab: they are never
    produced by `encode` (only prepended explicitly), so the BPE base and merges
    — and thus data/bpe.json — are unchanged by their presence."""
    global itos, stoi, vocab_size, source_ids, blend_id, end_id
    itos = list(base) + [a + b for a, b in merges] + SPECIAL_TOKENS
    stoi = {tok: i for i, tok in enumerate(itos)}
    vocab_size = len(itos)
    source_ids = {name[:-4]: stoi[tok] for name, tok in SOURCES.items()}  # stem -> id
    blend_id = stoi[BLEND_TOKEN]  # the joint-voice token, used when no source is chosen
    end_id = stoi[END_TOKEN]      # end-of-passage token
    rank = {tuple(pair): r for r, pair in enumerate(merges)}
    base_set = set(base)
    cache = {}

    def bpe_word(word):
        if word in cache:
            return cache[word]
        parts = [ch for ch in word if ch in base_set]  # unseen characters are dropped
        while len(parts) > 1:
            pairs = [(rank[p], j) for j, p in enumerate(zip(parts, parts[1:])) if p in rank]
            if not pairs:
                break
            _, j = min(pairs)  # apply the earliest-learned merge first, as in training
            parts[j:j + 2] = [parts[j] + parts[j + 1]]
        ids = [stoi[p] for p in parts]
        cache[word] = ids
        return ids

    def encode(text):
        text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
        return [i for w in WORD_RE.findall(text) for i in bpe_word(w)]

    globals()['encode'] = encode

# Define the model architecture, following GPT-2, blessed among the GPTs,
# Rotary position embeddings (RoPE): rotate each query/key head-dim pair by a
# position-dependent angle, so attention scores depend on relative position and
# no learned position table is needed.
_theta = 10000.0 ** (-torch.arange(0, n_embd // n_head, 2) / (n_embd // n_head))
_angles = torch.outer(torch.arange(block_size), _theta)
ROPE_COS, ROPE_SIN = _angles.cos().to(device), _angles.sin().to(device)

def rope(t, pos):
    cos, sin = ROPE_COS[pos], ROPE_SIN[pos]  # (T, head_dim/2), broadcast over batch and heads
    a, b = t.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)

# with microgpt's differences: layernorm -> rmsnorm, no biases, GeLU -> squared
# ReLU (the nanoGPT-speedrun finding), learned positions -> RoPE
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = nn.RMSNorm(n_embd)
        self.attn_qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.attn_wo = nn.Linear(n_embd, n_embd, bias=False)
        self.mlp_norm = nn.RMSNorm(n_embd)
        self.mlp_fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.mlp_fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x, pos, cache=None, use_cache=False):
        B, T, C = x.shape
        # 1) Multi-head Attention block
        q, k, v = self.attn_qkv(self.attn_norm(x)).split(n_embd, dim=2)
        q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v))
        q, k = rope(q, pos), rope(k, pos)  # cached keys were rotated when they were new
        if cache is not None:  # prepend the keys/values of all previous tokens
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        new_cache = (k, v) if use_cache else None  # build even on prefill (cache is None)
        # prefill (q covers all positions) needs the causal mask; a single decoded
        # token attends to everything before it, so no mask
        # note: no dropout inside attention — it would knock MPS off its fused
        # fast path and materialize the full T x T attention matrices
        y = F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[2] == k.shape[2])
        x = x + F.dropout(self.attn_wo(y.transpose(1, 2).reshape(B, T, C)), dropout, self.training)
        # 2) MLP block
        x = x + F.dropout(self.mlp_fc2(F.relu(self.mlp_fc1(self.mlp_norm(x))).square()), dropout, self.training)
        return x, new_cache

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)  # token embedding (positions come from RoPE)
        self.emb_norm = nn.RMSNorm(n_embd)
        self.blocks = nn.ModuleList(Block() for _ in range(n_layer))
        self.final_norm = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying, as in GPT-2
        # GPT-2-style init: Embedding's default N(0,1) is ~50x too large for
        # the tied lm_head and makes the initial loss explode
        nn.init.normal_(self.wte.weight, std=0.02)

    def forward(self, idx, caches=None):
        use_cache = caches is not None
        if not use_cache:
            caches = [None] * n_layer
        past = caches[0][0].shape[2] if caches[0] is not None else 0
        pos = torch.arange(past, past + idx.shape[1], device=idx.device)
        x = self.emb_norm(self.wte(idx))
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, pos, caches[i], use_cache)
            new_caches.append(c)
        logits = self.lm_head(self.final_norm(x))
        return (logits, new_caches) if use_cache else logits

# ---- Provenance and retrieval ---------------------------------------------
# The model composes wisdom in the canon's voice; it cannot be trusted to
# quote it. Retrieval closes that gap: every hit is a real passage, lifted
# verbatim from the source with a citation the reader can check. It is pure
# lexical search (BM25) over the corpus and needs no trained model at all.

WORKS = {  # filename -> (title, author) — author is '' for the Bible, cited by book
    'bible.txt':                 ('The King James Bible', ''),
    'meditations.txt':           ('Meditations', 'Marcus Aurelius'),
    'enchiridion.txt':           ('The Enchiridion', 'Epictetus'),
    'imitation_of_christ.txt':   ('The Imitation of Christ', 'Thomas a Kempis'),
    'confessions_augustine.txt': ('Confessions', 'St. Augustine'),
    'the_republic.txt':          ('The Republic', 'Plato'),
    'pilgrims_progress.txt':     ("The Pilgrim's Progress", 'John Bunyan'),
    'paradise_lost.txt':         ('Paradise Lost', 'John Milton'),
    'timaeus.txt':               ('Timaeus', 'Plato'),
    'boethius.txt':              ('The Consolation of Philosophy', 'Boethius'),
    'epictetus_discourses.txt':  ('Discourses', 'Epictetus'),
    'pascal_pensees.txt':        ('Pensées', 'Blaise Pascal'),
    'seneca_morals.txt':         ('Morals', 'Seneca'),
    'apocrypha.txt':             ('Wisdom of Solomon & Sirach', ''),  # cited by book+verse
    'hermetica.txt':             ('The Divine Pymander', 'Hermes Trismegistus'),
    'kybalion.txt':              ('The Kybalion', 'Three Initiates'),
    'apology.txt':               ('Apology', 'Plato'),
    'phaedo.txt':                ('Phaedo', 'Plato'),
    'symposium.txt':             ('Symposium', 'Plato'),
    'nicomachean_ethics.txt':    ('Nicomachean Ethics', 'Aristotle'),
    'summa_theologica.txt':      ('Summa Theologica I', 'Thomas Aquinas'),
    'kierkegaard.txt':           ('Selections', 'Søren Kierkegaard'),
    'heraclitus.txt':            ('Fragments', 'Heraclitus'),
    'plotinus.txt':              ('The Enneads', 'Plotinus'),
    'cicero.txt':                ('Philosophical Treatises', 'Cicero'),
    'iamblichus.txt':            ('On the Mysteries', 'Iamblichus'),
    'golden_verses.txt':         ('The Golden Verses of Pythagoras', "Fabre d'Olivet"),
    'lucretius.txt':             ('On the Nature of Things', 'Lucretius'),
    'philo.txt':                 ('The Allegorical Commentary', 'Philo of Alexandria'),
    'justin_martyr.txt':         ('Apologies & the Dialogue with Trypho', 'Justin Martyr'),
    'athanasius.txt':            ('On the Incarnation of the Word', 'Athanasius'),
    'clement_alexandria.txt':    ('Exhortation, Instructor & Stromata', 'Clement of Alexandria'),
    'cratylus.txt':              ('Cratylus', 'Plato'),
    'phaedrus.txt':              ('Phaedrus', 'Plato'),
    'theaetetus.txt':            ('Theaetetus', 'Plato'),
    'sophist.txt':               ('Sophist', 'Plato'),
    'parmenides.txt':            ('Parmenides', 'Plato'),
    'gorgias.txt':               ('Gorgias', 'Plato'),
    'meno.txt':                  ('Meno', 'Plato'),
    'philebus.txt':              ('Philebus', 'Plato'),
    'dante.txt':                 ('The Divine Comedy', 'Dante Alighieri'),
}

# stem -> human label for the voice selector, e.g. 'meditations' -> 'Marcus Aurelius — Meditations'
VOICES = {name[:-4]: (f'{author} — {title}' if author else title)
          for name, (title, author) in WORKS.items()}

# The 66 books of the KJV in order. The Gutenberg text marks the first verse of
# each book with a bare "1:1" (chapter headings are multi-line and aliased, e.g.
# "The First Book of Samuel / Otherwise Called: The First Book of the Kings"),
# so we hand out these names in sequence instead of parsing the headings.
BIBLE_BOOKS = [
    'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 'Joshua',
    'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings',
    '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther', 'Job',
    'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon', 'Isaiah',
    'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos',
    'Obadiah', 'Jonah', 'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai',
    'Zechariah', 'Malachi', 'Matthew', 'Mark', 'Luke', 'John', 'Acts',
    'Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians',
    'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians',
    '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews', 'James',
    '1 Peter', '2 Peter', '1 John', '2 John', '3 John', 'Jude', 'Revelation',
]
VERSE_RE = re.compile(r'(\d+):(\d+) ')
APOCRYPHA_BOOKS = ['Wisdom of Solomon', 'Sirach']  # order of the two wisdom books in the slice

def parse_versed(text, books, work):
    """Split a chapter:verse scripture into one citable passage per verse. Books are
    handed out in order, advancing on each '1:1' (the sole such marker per book)."""
    out, cur, book = [], None, -1
    for line in text.split('\n'):
        marks = list(VERSE_RE.finditer(line))
        if not marks:
            if line.lstrip().startswith('['):  # a bracketed section heading (Apocrypha)
                cur = None                      # ends the verse; drops any prologue that follows
            elif cur and line.strip():          # else a numberless line continues the verse above
                cur['text'] += ' ' + line.strip()
            continue
        lead = line[:marks[0].start()].strip()
        if lead and cur:  # text before the first number belongs to the previous verse
            cur['text'] += ' ' + lead
        for i, m in enumerate(marks):  # a line may pack several verses ("1:14 ... 1:15 ...")
            chap, verse = int(m[1]), int(m[2])
            end = marks[i + 1].start() if i + 1 < len(marks) else len(line)
            if chap == 1 and verse == 1:  # the only "1:1" per book marks a new book
                book += 1
            name = books[book] if 0 <= book < len(books) else '?'
            cur = {'work': work, 'ref': f'{name} {chap}:{verse}',
                   'text': line[m.end():end].strip()}
            out.append(cur)
    return out

def is_canonical(p):
    """Reject titles, running heads, tables of contents, and editorial matter."""
    if len(p) < 30 or len(p.split()) < 6:
        return False
    letters = [c for c in p if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        return False  # an ALL-CAPS heading, not a sentence
    return not re.match(r'(Produced by|Translated by|Transcrib|Contents|Note[:s]|'
                        r'Footnote|Illustration|Chapter |CHAPTER |\[)', p)

def chunk_words(p, target=55, hard=120):
    """Keep normal paragraphs whole; split over-long ones on sentence boundaries
    so a hit is a quotable passage, not a whole chapter of Paradise Lost."""
    if len(p.split()) <= hard:
        return [p]
    chunks, buf, n = [], [], 0
    for sent in re.split(r'(?<=[.!?;:]) +', p):
        buf.append(sent)
        n += len(sent.split())
        if n >= target:
            chunks.append(' '.join(buf))
            buf, n = [], 0
    if buf:
        tail = ' '.join(buf)
        if chunks:
            chunks[-1] += ' ' + tail
        else:
            chunks.append(tail)
    return chunks

def parse_prose(text, title, author):
    """Split a prose or verse work into quotable passages, skipping front matter."""
    out, n = [], 0
    for para in text.split('\n'):
        para = para.strip()
        if not is_canonical(para):
            continue
        for chunk in chunk_words(para):
            n += 1
            out.append({'work': author or title, 'ref': f'{title} §{n}', 'text': chunk})
    return out

def load_passages():
    """Fetch the canon (downloading if needed) and return every citable passage."""
    os.makedirs('data', exist_ok=True)
    passages = []
    for name, (title, author) in WORKS.items():
        text = fetch_book(name, BOOKS[name])
        if name == 'bible.txt':
            passages += parse_versed(text, BIBLE_BOOKS, 'King James Bible')
        elif name == 'apocrypha.txt':
            passages += parse_versed(text, APOCRYPHA_BOOKS, 'Apocrypha (KJV)')
        else:
            passages += parse_prose(text, title, author)
    return passages

class Bm25:
    """A tiny Okapi BM25 index: lexical search over the canon, no model needed."""
    def __init__(self, passages, k1=1.5, b=0.75):
        self.passages, self.k1, self.b = passages, k1, b
        self.postings = defaultdict(list)  # term -> [(passage index, term frequency)]
        self.doc_len, df = [], Counter()
        for d, p in enumerate(passages):
            tf = Counter(self.tokens(p['text']))
            self.doc_len.append(sum(tf.values()))
            for term, f in tf.items():
                self.postings[term].append((d, f))
                df[term] += 1
        self.N = max(len(passages), 1)
        self.avgdl = sum(self.doc_len) / self.N
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    @staticmethod
    def tokens(s):
        return re.findall(r"[a-z']+", s.lower())

    def search(self, query, k=6):
        scores = defaultdict(float)
        for term in set(self.tokens(query)):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for d, f in self.postings[term]:
                dl = self.doc_len[d] or 1
                scores[d] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        ranked = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
        return [(self.passages[d], round(s, 3)) for d, s in ranked]

if search_mode:
    print("indexing the canon for retrieval ...")
    index = Bm25(load_passages())
    print(f"indexed {len(index.passages):,} passages from {len(WORKS)} works")

    def show(query):
        hits = index.search(query)
        if not hits:
            print("  (the canon holds no such words)")
        for p, score in hits:
            print(f"\n  \033[1m{p['ref']}\033[0m  \033[2m{p['work']} · {score}\033[0m")
            print(textwrap.fill(p['text'], width=78, initial_indent='    ', subsequent_indent='    '))

    tail = [a for a in sys.argv[sys.argv.index('search') + 1:] if not a.startswith('--')]
    if tail:  # one-shot: python LogosGPT.py search "a still small voice"
        show(' '.join(tail))
    else:  # interactive
        print("\n--- search the canon --- (type a query, Ctrl-D to exit)")
        while True:
            try:
                q = input("\nseek> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q:
                show(q)
    sys.exit(0)

# When sampling or serving, rebuild the exact architecture and tokenizer the
# checkpoint was trained with; otherwise build the corpus and train
checkpoint = None
if sample_only and os.path.exists('logosgpt.pt'):
    checkpoint = torch.load('logosgpt.pt', map_location=device)
    globals().update(checkpoint['config'])
    make_tokenizer(checkpoint['bpe_base'], checkpoint['bpe_merges'])
else:
    os.makedirs('data', exist_ok=True)
    book_texts = {name: fetch_book(name, url) for name, url in BOOKS.items()}
    for name, t in book_texts.items():
        assert ''.join(WORD_RE.findall(t)) == t, f"pre-tokenizer does not cover {name}"
        print(f"  {name:28s} {len(t):>9,} chars")

    # Let there be a Tokenizer, learned from the corpus itself (cached on disk;
    # delete data/bpe.json after changing BOOKS or vocab_target)
    if os.path.exists('data/bpe.json'):
        bpe = json.load(open('data/bpe.json'))
    else:
        print(f"training BPE tokenizer, target vocab {vocab_target:,} ...")
        word_freq = Counter(w for t in book_texts.values() for w in WORD_RE.findall(t))
        base, merges = train_bpe(word_freq, vocab_target)
        bpe = {'base': base, 'merges': merges}
        json.dump(bpe, open('data/bpe.json', 'w'))
    make_tokenizer(bpe['base'], [tuple(m) for m in bpe['merges']])

    # Hold out the last 5% of every book (not just the last book) for validation.
    # Keep each book's tokens separate — get_batch samples a window from one book
    # and tags it with that book's source token, so books are never spliced.
    train_books, val_books = [], []  # each: (source id, 1-D LongTensor of token ids)
    total, newline = 0, stoi['\n']
    for name, t in book_texts.items():
        # every book is one passage per line, so each newline is a passage end:
        # relabel it <end>, the boundary the model learns to stop on
        ids = [end_id if x == newline else x for x in encode(t)]
        total += len(ids)
        k = int(0.95 * len(ids))
        src = source_ids[name[:-4]]
        train_books.append((src, torch.tensor(ids[:k], dtype=torch.long)))
        if len(ids) - k > block_size:  # hold out only books long enough to sample from
            val_books.append((src, torch.tensor(ids[k:], dtype=torch.long)))
    chars_per_token = sum(len(t) for t in book_texts.values()) / total
    print(f"corpus: {total:,} tokens | vocab size: {vocab_size:,} "
          f"| {len(SOURCE_TOKENS)} sources + <canon> | {chars_per_token:.2f} chars/token")

model = GPT().to(device)
print(f"num params: {sum(p.numel() for p in model.parameters()):,} (embeddings tied)")

# Repeat in sequence (unless we are only here to sample from a saved model)
if checkpoint is not None:
    model.load_state_dict(checkpoint['model'])
else:
    def get_batch(split):
        # pick books in proportion to their length (so KJV still dominates), then
        # a random window from each; prepend the source token as x[:, 0] and shift
        # so the model predicts the book's own next token from position 0 onward.
        books = train_books if split == 'train' else val_books
        weights = torch.tensor([len(ids) for _, ids in books], dtype=torch.float)
        picks = torch.multinomial(weights, batch_size, replacement=True).tolist()
        x = torch.empty(batch_size, block_size, dtype=torch.long)
        y = torch.empty(batch_size, block_size, dtype=torch.long)
        for row, b in enumerate(picks):
            src, ids = books[b]
            i = torch.randint(len(ids) - block_size + 1, (1,)).item()
            seg = ids[i:i + block_size]  # block_size real tokens of this book
            # tag with the book's source, or <canon> a fraction of the time so the
            # joint voice is trained on the marginal over all books, not left OOD
            x[row, 0] = blend_id if torch.rand(1).item() < blend_dropout else src
            x[row, 1:] = seg[:-1]
            y[row] = seg
        return x.to(device), y.to(device)

    @torch.no_grad()
    def val_loss(num_batches=20):
        model.eval()
        losses = torch.zeros(num_batches)
        for i in range(num_batches):
            x, y = get_batch('val')
            logits = model(x)
            losses[i] = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        model.train()
        return losses.mean().item()

    # weight decay on the matrices (embeddings, attention, MLP) but not rmsnorm gains
    optimizer = torch.optim.AdamW(
        [{'params': [p for p in model.parameters() if p.dim() >= 2], 'weight_decay': weight_decay},
         {'params': [p for p in model.parameters() if p.dim() < 2], 'weight_decay': 0.0}],
        lr=learning_rate)

    def save_checkpoint():
        torch.save({
            'config': {k: globals()[k] for k in ('n_layer', 'n_embd', 'n_head', 'block_size')},
            'bpe_base': bpe['base'],
            'bpe_merges': [tuple(m) for m in bpe['merges']],
            'model': model.state_dict(),
        }, 'logosgpt.pt')

    best_val = float('inf')
    t0 = time.time()
    for step in range(num_steps):
        # warmup then cosine decay to min_lr: the ramp steadies the fresh weights,
        # the cosine anneal lets the model settle into the loss basin near the end
        if step < warmup_steps:
            lr = learning_rate * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
            lr = min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = lr

        # Forward a batch of passages, compute the average next-token loss
        x, y = get_batch('train')
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        # Backward the loss and update the parameters. May yours be low.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0 or step == 0:
            print(f"step {step+1:5d} / {num_steps} | train loss {loss.item():.4f} | {time.time()-t0:.0f}s")
        if (step + 1) % eval_every == 0:
            vl = val_loss()
            saved = ''
            if vl < best_val:  # keep the best-val checkpoint: long runs can only help
                best_val = vl
                save_checkpoint()
                saved = ' | saved'
            print(f"step {step+1:5d} / {num_steps} | val loss   {vl:.4f}{saved}")

    if best_val == float('inf'):  # run too short to ever evaluate: save what we have
        save_checkpoint()
        print("saved model to logosgpt.pt")
    else:
        print(f"saved model to logosgpt.pt (best val loss {best_val:.4f})")

# Inference: may the model babble wisdom back to us
model.eval()
CONTROL_ID_TENSOR = torch.tensor([stoi[t] for t in CONTROL_TOKENS], device=device)  # never output

@torch.no_grad()
def generate_stream(prompt, max_new_tokens=180, temperature=0.8, source=None, top_k=40):
    # A known voice stem ('meditations') leads the context with its token; anything
    # else (None, 'canon', 'blend') leads with <canon>, the joint voice trained over
    # the whole corpus. The lead token is kept in front across cache resets, so the
    # chosen voice — or the joint one — persists past the context window. Generation
    # runs until the model emits <end> (a whole passage) or hits max_new_tokens.
    lead = source_ids[source] if source in source_ids else blend_id
    room = block_size - 1  # one slot always goes to the lead token

    def prefill(body):  # fresh forward over [lead token] + the most recent tokens
        seq = [lead] + body[-room:]
        return model(torch.tensor([seq], device=device), caches=[None] * n_layer)

    body = encode(prompt)[-room:]
    if not body:
        # No prompt: seed with <end>. Training windows start at random offsets, so the
        # lead token alone would continue from mid-sentence; what follows an <end> (a
        # line break in training) is always a fresh paragraph, so this opens a passage.
        body = [end_id]
    logits, caches = prefill(body)
    for i in range(max_new_tokens):
        step = logits[:, -1, :].clone()
        step[:, CONTROL_ID_TENSOR] = -float('inf')  # a source/canon token is never valid output
        if i == 0:
            step[:, end_id] = -float('inf')  # a passage may not end before it begins
        if top_k and top_k < step.size(-1):
            # keep only the k likeliest tokens: a small model puts real mass on a long
            # garbage tail, and one bad pick derails the whole passage. Truncating first
            # is the biggest single lever on coherence at inference time.
            kth = torch.topk(step, top_k, dim=-1).values[:, -1:]
            step[step < kth] = -float('inf')
        next_id = torch.multinomial(F.softmax(step / temperature, dim=-1), num_samples=1)
        if next_id.item() == end_id:  # the model has finished a passage
            break
        yield itos[next_id.item()]
        body.append(next_id.item())
        if 1 + len(body) >= block_size:
            # the position table is full: restart the cache from the recent half
            # (the O(T) re-encode happens once per ~64 tokens), lead token intact
            body = body[-(block_size // 2):]
            logits, caches = prefill(body)
        else:
            logits, caches = model(next_id, caches)

def generate(prompt, **kwargs):
    return ''.join(generate_stream(prompt, **kwargs))

if chat_mode:
    # This is a base model, not a chatbot: it only knows how to continue text.
    print("\n--- passage completion --- (type the start of a passage, Ctrl-D to exit)")
    print(f"voices: {', '.join(source_ids)}  (type ':<voice>' to choose one, ':canon' for the joint voice)")
    voice = None  # None = <canon>, the joint voice trained over the whole corpus
    while True:
        try:
            prompt = input(f"\npassage [{voice or 'blend'}]> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.startswith(':'):  # a voice command, not a passage
            choice = prompt[1:].strip()
            if choice in ('', 'blend', 'none', 'canon'):
                voice = None
            elif choice in source_ids:
                voice = choice
            else:
                print(f"  unknown voice; choose from: {', '.join(source_ids)}")
            continue
        print(f"LogosGPT: {prompt}", end='', flush=True)
        for piece in generate_stream(prompt, source=voice):
            print(piece, end='', flush=True)
        print()

elif web_mode:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    print("indexing the canon for retrieval ...")
    retrieval = Bm25(load_passages())
    print(f"indexed {len(retrieval.passages):,} passages from {len(WORKS)} works")

    stats = f"{sum(p.numel() for p in model.parameters()):,} parameters · {vocab_size:,}-token BPE vocabulary · trained on {len(BOOKS)} books · {len(retrieval.passages):,} passages indexed"
    voice_opts = '<option value="">All voices — the blend</option>' + ''.join(
        f'<option value="{stem}">{label}</option>' for stem, label in VOICES.items())
    PAGE = open('logos_ui.html').read().replace('__STATS__', stats).replace('__VOICES__', voice_opts)
    gen_lock = threading.Lock()  # one generation at a time on the little GPU

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path != '/':
                self.send_error(404)
                return
            body = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path not in ('/generate', '/search'):
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/search':
                # retrieval: return real, cited passages — no model, no invention
                hits = retrieval.search(str(req.get('query', ''))[:200])
                body = json.dumps([{'ref': p['ref'], 'work': p['work'],
                                    'text': p['text'], 'score': s} for p, s in hits]).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            prompt = str(req.get('prompt', ''))[:2000]
            temperature = min(max(float(req.get('temperature', 0.8)), 0.1), 2.0)
            max_new = min(max(int(req.get('max_tokens', 180)), 1), 600)
            top_k = min(max(int(req.get('top_k', 40)), 1), vocab_size)
            source = str(req.get('source', '')) or None  # a voice stem, or None for the blend
            if source not in source_ids:
                source = None
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Connection', 'close')
            self.end_headers()
            try:
                with gen_lock:
                    for piece in generate_stream(prompt, max_new_tokens=max_new,
                                                 temperature=temperature, source=source, top_k=top_k):
                        self.wfile.write(piece.encode())
                        self.wfile.flush()
            except BrokenPipeError:
                pass  # reader closed the tab or pressed Stop

    addr = ('127.0.0.1', 8000)
    print(f"\nLogosGPT is listening on http://{addr[0]}:{addr[1]} (Ctrl-C to stop)")
    ThreadingHTTPServer(addr, Handler).serve_forever()

else:
    print("\n--- inference (new, hallucinated wisdom) ---")
    print(generate('', max_new_tokens=300).strip())
