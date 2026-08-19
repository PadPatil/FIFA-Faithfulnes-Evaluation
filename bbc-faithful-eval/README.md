# faithful-cuts

## Running in Colab

Open the notebook via the Colab badge at the top of `CSED504_Faithful_Cuts.ipynb`.
The first code cell clones this repo into the Colab session and adds the cloned
directory to `sys.path` so `import utils_video2text` / `import parse_utils` resolve
against the checked-out files. The current working directory stays at `/content`;
the TIFA160 CSV is loaded via a `__file__`-relative path inside
`utils_video2text.py`, so nothing depends on `%cd`-ing into the clone.

If you rearrange the notebook, keep the clone-and-`sys.path.insert` cell before
any import of `utils_video2text` or `parse_utils`.
