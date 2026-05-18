#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/inspire/qb-ilm2/project/26summer-camp-01/26210300"
ENV_SCRIPT="$PROJECT_ROOT/env/inspire-global-dirs.sh"
BASHRC="${HOME}/.bashrc"
MARKER_BEGIN="# >>> inspire-global-dirs >>>"
MARKER_END="# <<< inspire-global-dirs <<<"

if [ ! -f "$ENV_SCRIPT" ]; then
  echo "Missing env script: $ENV_SCRIPT" >&2
  exit 1
fi

bash "$ENV_SCRIPT"

if [ -f "$BASHRC" ] && grep -Fq "$MARKER_BEGIN" "$BASHRC"; then
  echo "Global dir config already exists in $BASHRC"
  exit 0
fi

cat >> "$BASHRC" <<EOF

$MARKER_BEGIN
if [ -f "$ENV_SCRIPT" ]; then
  source "$ENV_SCRIPT"
fi
$MARKER_END
EOF

echo "Appended global dir config to $BASHRC"
