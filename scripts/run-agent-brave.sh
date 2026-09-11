#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

npm run build

run_dir="$(mktemp -d "${TMPDIR:-/tmp}/wir-agent-brave.XXXXXX")"
profile_dir="$run_dir/profile"
cdp_port="${WIR_CDP_PORT:-9222}"
start_url="https://google.com"
brave_app="${BRAVE_APP:-$(mdfind "kMDItemCFBundleIdentifier == 'com.brave.Browser'" | head -n 1)}"

if [[ -z "$brave_app" || ! -d "$brave_app" ]]; then
  echo "Brave Browser.app was not found by macOS Spotlight (mdfind)." >&2
  exit 1
fi

brave_bin="$brave_app/Contents/MacOS/Brave Browser"
if [[ ! -x "$brave_bin" ]]; then
  echo "Brave executable not found: $brave_bin" >&2
  exit 1
fi

echo "Brave: $brave_app" >&2
echo "Agent artifacts: $run_dir" >&2

"$brave_bin" \
  --remote-debugging-port="$cdp_port" \
  --user-data-dir="$profile_dir" \
  --no-first-run \
  --no-default-browser-check \
  "$start_url" >"$run_dir/brave.log" 2>&1 &
brave_pid=$!
trap 'kill "$brave_pid" 2>/dev/null || true' EXIT

for _ in {1..60}; do
  if curl --silent --fail "http://127.0.0.1:$cdp_port/json/version" >/dev/null; then
    break
  fi
  sleep 0.25
done

if ! curl --silent --fail "http://127.0.0.1:$cdp_port/json/version" >/dev/null; then
  echo "Brave CDP endpoint did not become ready: http://127.0.0.1:$cdp_port" >&2
  exit 1
fi

node dist/agent/main.js <<JSON
{
  "runId": "brave-foreground-example",
  "task": {
    "benchmark": "local-example",
    "taskId": 0,
    "revision": 1,
    "instruction": "Read the page title and finish with it.",
    "startUrl": "$start_url",
    "expectedAction": "RETRIEVE"
  },
  "authority": {
    "riskMode": "read_only",
    "originPolicy": "observed"
  },
  "browser": {
    "headless": false,
    "cdpEndpoint": "http://127.0.0.1:$cdp_port",
    "harPath": "$run_dir/network.har",
    "tracePath": "$run_dir/trace.zip",
    "storageStatePath": null,
    "slowMoMs": 0
  },
  "budgets": {
    "maxWallTimeMs": 120000,
    "maxModelCalls": 8,
    "maxWirCalls": 20
  },
  "artifacts": {
    "trajectoryPath": "$run_dir/trajectory.jsonl",
    "metricsPath": "$run_dir/metrics.json"
  }
}
JSON
