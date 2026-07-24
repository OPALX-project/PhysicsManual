The complete physics manual is now part of the [unified manual](https://github.com/OPALX-project/opalx-manual) and will not be a separate repository anymore. 

# PhysicsManual

Describes the underlying physics of OPALX.

## Render

Render the full manual:

```bash
quarto render /Users/adelmann/git/physics-manual-opalx --to html
```

Render one chapter to PDF:

```bash
quarto render /Users/adelmann/git/physics-manual-opalx/sections/coordinate-systems/index.qmd --to pdf
quarto render /Users/adelmann/git/physics-manual-opalx/sections/gamma-gamma/index.qmd --to pdf
```

The website output is written to `docs/`.

## Publish Workflow

The repository uses two branches for different purposes:

- `main` contains the Quarto source files, figures, bibliography, scripts, and
  workflow configuration.
- `gh-pages` contains the generated website that GitHub Pages serves.

Normal editing workflow:

```bash
git checkout main
quarto render . --to html
git add _quarto.yml index.qmd sections references.bib README.md scripts
git commit -m "Describe the manual change"
git push origin main
```

After `main` is pushed, the GitHub Actions workflow renders the HTML manual,
builds the OPALX Doxygen API, and publishes the generated site to `gh-pages`.
Do not manually commit or push `docs/` from `main`; it is generated output and
is ignored there.

To check the deployed output, inspect the live GitHub Pages site or the
generated files on `gh-pages`. Do not edit `gh-pages` directly in normal use.

## OPALX Doxygen API

Build the local OPALX API documentation after rendering the HTML manual:

```bash
./scripts/build-opalx-doxygen.sh
```

By default the script clones `https://github.com/OPALX-project/OPALX.git` at
`master` into `.external/OPALX` and writes Doxygen HTML to
`docs/api/opalx/html`. Override `OPALX_REF`, `OPALX_SRC_DIR`, or
`OPALX_DOXYGEN_OUTPUT` when you need a different OPALX checkout or output path.
Manual links use `/api/opalx/html/...`, so the API pages must be present under
`docs/api/opalx/html` before publishing.

## TikZ Workflow

For the `gamma-gamma` notes, TikZ is handled with one source of truth and two output paths:

- `figures/tikz/*.tikz.tex` contains the actual TikZ drawing code.
- `figures/*.tex` are thin standalone wrappers that `\input{...}` the TikZ snippets.
- the Quarto chapters use rendered images for HTML and raw TikZ for PDF.
- `includes/tikz-preamble.tex` is loaded globally for Quarto PDF output from `_quarto.yml`.

Current source files:

- `sections/gamma-gamma/figures/tikz/ics_90deg_figure.tikz.tex`
- `sections/gamma-gamma/figures/tikz/breit_wheeler_qed_style_tikz.tikz.tex`

Current wrappers:

- `sections/gamma-gamma/figures/ics_90deg_figure.tex`
- `sections/gamma-gamma/figures/breit_wheeler_qed_style_tikz.tex`

Current chapter usage:

- `sections/gamma-gamma/index.qmd`
- `sections/gamma-gamma/_linear-compton-benchmark.qmd`
- `sections/gamma-gamma/_BreitWheeler.qmd`

### Edit a TikZ Figure

1. Edit the snippet in `sections/gamma-gamma/figures/tikz/`.
2. If the HTML fallback image also needs to change, regenerate the figure assets from the wrapper in `sections/gamma-gamma/figures/`.
3. Re-render the affected Quarto page.

### HTML vs PDF

- HTML does not render raw TikZ directly, so the chapters use image files such as `png`.
- PDF renders the same figure through raw LaTeX using `\input{figures/tikz/...}`.

This means the TikZ snippet is the canonical source, while HTML uses a pre-rendered fallback.
