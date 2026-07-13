# LogosGPT

A small (~16M-parameter) GPT over a canon of scripture and Western wisdom,
chosen around the theme of the **Logos** (the Word / ordering reason) and
**Truth** — the King James Bible (with the Wisdom-of-Solomon/Sirach Apocrypha),
Heraclitus, Plato (Republic, Timaeus, Apology, Phaedo, Symposium), Aristotle,
Cicero, Lucretius, Marcus Aurelius, Epictetus, Seneca, Plotinus, Iamblichus,
the Golden Verses of Pythagoras, Augustine, Boethius, Aquinas, Thomas à Kempis,
Pascal, Bunyan, Milton, Kierkegaard, the Corpus Hermeticum, and the Kybalion.
It runs entirely on your machine and does two distinct things:

- **Generation** — composes new text in a chosen voice. This is *invention*, not
  quotation: nothing it writes is a real passage.
- **Retrieval** — searches the corpus and returns *real* passages verbatim, each
  with a citation you can check (e.g. `Genesis 1:1`). No model required.

## Setup

```bash
pip install -r requirements.txt        # only dependency: torch >= 2.4
```

The first run downloads the corpus into `data/` (Project Gutenberg, plus the
Corpus Hermeticum from Wikisource) and trains the BPE tokenizer on it.

## Commands

```bash
python LogosGPT.py                 # train from scratch, then sample; saves logosgpt.pt
python LogosGPT.py --steps 200     # quick smoke test (short training run)

python LogosGPT.py sample          # generate from the saved model
python LogosGPT.py chat            # interactive passage completion
python LogosGPT.py search "mercy"  # retrieve real, cited passages (no model needed)
python LogosGPT.py web             # serve the web UI at http://127.0.0.1:8000
```

`sample`, `chat`, and `web` need a trained `logosgpt.pt` (run `python LogosGPT.py`
first). `search` works on the corpus alone.

## Hardware

The default config (8 layers, 384 wide, 512-token context, batch 128) is sized
for ~48GB of unified memory — roughly a 4-hour run on an M5 Pro. Training
evaluates every 250 steps and keeps the **best-validation checkpoint**, so long
runs can't overfit the saved model and interrupting mid-run is safe.

On a small machine (8GB), set `batch_size = 8` at the top of `LogosGPT.py`.

## Voices

Generation can be steered to a single source, or left to the joint voice:

```
bible · meditations · enchiridion · imitation_of_christ · confessions_augustine
the_republic · pilgrims_progress · paradise_lost · timaeus · boethius
epictetus_discourses · pascal_pensees · seneca_morals · apocrypha · hermetica
kybalion · apology · phaedo · symposium · nicomachean_ethics · summa_theologica
kierkegaard · heraclitus · plotinus · cicero · iamblichus · golden_verses
lucretius · canon (all sources jointly)
```

- **chat:** type `:meditations` to pick a voice, `:canon` for the joint voice.
- **web:** choose from the *Voice* dropdown.

Every generated passage ends when the model emits its `<end>` token — trained
at real passage boundaries — so responses stop at a natural close instead of a
fixed length.

## Files

| Path             | What it is                                          |
| ---------------- | --------------------------------------------------- |
| `LogosGPT.py`    | model, tokenizer, training, retrieval, CLI + server |
| `logos_ui.html`  | the web UI                                          |
| `data/`          | cached corpus + BPE tokenizer (`bpe.json`)          |
| `logosgpt.pt`    | trained checkpoint (created by training)            |
