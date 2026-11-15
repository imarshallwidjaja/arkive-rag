#!/bin/sh
# run-scenario MODEL PROMPT: answer one prompt with OpenCode and print its JSON event stream.
#
# `opencode run --standalone` sends the first model request without waiting for local MCP
# servers to connect, so a server that needs a few seconds to start (arkive-rag opens every
# snapshot and its embedding matrix) is missing from the model's tool list. This script starts
# a private `opencode serve`, waits until the `arkive` server reports connected, and only then
# runs the prompt against that server. On a loaded machine the server can take tens of seconds
# to start, and OpenCode never retries a server that missed its startup timeout, so
# opencode.json sets that timeout to 180 s and this script waits up to 240 s, which also
# covers `opencode serve` starting. Exit 3: the MCP server failed or did not connect in time.
#
# The container runs as the invoking host user, which has no account or home in the image,
# so HOME and the working directory live under /tmp.
set -eu

model=$1
prompt=$2
server=http://127.0.0.1:4096
HOME=/tmp/home
export HOME
mkdir -p "$HOME/work"
cd "$HOME/work"
OPENCODE_SERVER_PASSWORD=$(head -c 24 /dev/urandom | base64)
OPENCODE_PASSWORD=$OPENCODE_SERVER_PASSWORD
export OPENCODE_SERVER_PASSWORD OPENCODE_PASSWORD

opencode serve --port 4096 --hostname 127.0.0.1 >/tmp/opencode-serve.log 2>&1 &

# Each `opencode mcp list` is bounded, so a stalled call cannot hold the loop past the
# deadline by more than one bound.
mcp_list() {
    timeout -s KILL 10 opencode mcp list --server "$server"
}

started=$(date +%s)
deadline=$((started + 240))
while :; do
    status=$(mcp_list 2>/dev/null | grep 'arkive ' || true)
    case $status in
    *connected*) break ;;
    *failed*) problem="failed after $(($(date +%s) - started)) s" ;;
    *) [ "$(date +%s)" -lt "$deadline" ] && { sleep 1; continue; }
       problem="did not connect within 240 s" ;;
    esac
    echo "arkive MCP server $problem:" >&2
    mcp_list >&2 || true
    tail -n 20 /tmp/opencode-serve.log >&2
    exit 3
done

# `--` ends the options, so a prompt that begins with a dash stays the message.
exec opencode run --server "$server" --model "$model" --format json --auto -- "$prompt"
