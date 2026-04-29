#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OPALX_REPO_URL="${OPALX_REPO_URL:-https://github.com/OPALX-project/OPALX.git}"
OPALX_REF="${OPALX_REF:-master}"
OPALX_SRC_DIR="${OPALX_SRC_DIR:-$ROOT_DIR/.external/OPALX}"
OPALX_DOXYGEN_OUTPUT="${OPALX_DOXYGEN_OUTPUT:-$ROOT_DIR/docs/api/opalx}"

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required to fetch OPALX" >&2
  exit 1
fi

if ! command -v doxygen >/dev/null 2>&1; then
  echo "error: doxygen is required to build OPALX API documentation" >&2
  exit 1
fi

if [ ! -d "$OPALX_SRC_DIR/.git" ]; then
  mkdir -p "$(dirname "$OPALX_SRC_DIR")"
  git clone --depth 1 --branch "$OPALX_REF" "$OPALX_REPO_URL" "$OPALX_SRC_DIR"
else
  git -C "$OPALX_SRC_DIR" fetch --depth 1 origin "$OPALX_REF"
  git -C "$OPALX_SRC_DIR" checkout --detach FETCH_HEAD
fi

if [ ! -f "$OPALX_SRC_DIR/Doxyfile.in" ]; then
  echo "error: OPALX Doxyfile.in was not found in $OPALX_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OPALX_DOXYGEN_OUTPUT"
rm -rf "$OPALX_DOXYGEN_OUTPUT/html"

OPALX_COMMIT="$(git -C "$OPALX_SRC_DIR" rev-parse --short=12 HEAD)"
DOXYFILE="$OPALX_DOXYGEN_OUTPUT/Doxyfile.physics-manual"

{
  printf '@INCLUDE = %s\n' "$OPALX_SRC_DIR/Doxyfile.in"
  printf 'PROJECT_NUMBER = %s (%s)\n' "$OPALX_REF" "$OPALX_COMMIT"
  printf 'OUTPUT_DIRECTORY = %s\n' "$OPALX_DOXYGEN_OUTPUT"
  printf 'HTML_OUTPUT = html\n'
  printf 'INPUT = %s/src %s/unit_tests\n' "$OPALX_SRC_DIR" "$OPALX_SRC_DIR"
  printf 'STRIP_FROM_PATH = %s\n' "$OPALX_SRC_DIR"
  printf 'WARN_LOGFILE = %s\n' "$OPALX_DOXYGEN_OUTPUT/doxygen-warnings.log"
  printf 'CREATE_SUBDIRS = NO\n'
  printf 'SHORT_NAMES = NO\n'
  printf 'GENERATE_TREEVIEW = YES\n'
  printf 'SEARCHENGINE = YES\n'
  printf 'QUIET = YES\n'
} > "$DOXYFILE"

doxygen "$DOXYFILE"

cat > "$OPALX_DOXYGEN_OUTPUT/README.md" <<EOF
# OPALX Doxygen API

Generated from \`$OPALX_REPO_URL\` at \`$OPALX_REF\` (\`$OPALX_COMMIT\`).

Open \`html/index.html\` or use the Physics Manual links under \`/api/opalx/html/\`.
EOF

echo "Built OPALX Doxygen HTML in $OPALX_DOXYGEN_OUTPUT/html"
