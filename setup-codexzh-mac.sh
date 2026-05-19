#!/bin/bash

set -e

API_KEY="sk-wFNMlGQLA48osxpa3UtKYxh5csTp3d0tYlhqpIar7Elo4byM"
if [ -z "$API_KEY" ] || [ "$API_KEY" = "__API_KEY__" ]; then
  echo "ERROR: API Key not set"
  exit 1
fi


CODEX_DIR="$HOME/.codex"
mkdir -p "$CODEX_DIR"

cat > "$CODEX_DIR/config.toml" << 'EOF'
model_provider = "codexzh"
model = "gpt-5.2"
model_reasoning_effort = "high"
disable_response_storage = false

[model_providers.codexzh]
name = "codexzh"
base_url = "https://chat.ekti.cc/v1"
wire_api = "responses"
requires_openai_auth = true
web_search = "live"

EOF

cat > "$CODEX_DIR/auth.json" << EOF
{
  "OPENAI_API_KEY": "$API_KEY"
}
EOF

chmod 600 "$CODEX_DIR/config.toml" "$CODEX_DIR/auth.json" || true

echo "OK: Files written to $CODEX_DIR"
