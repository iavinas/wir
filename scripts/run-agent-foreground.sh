#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
npm run build

run_dir="$(mktemp -d "${TMPDIR:-/tmp}/wir-agent-example.XXXXXX")"
echo "Agent artifacts: $run_dir" >&2

node dist/agent/main.js <<JSON
{
  "runId": "foreground-example",
  "task": {
    "benchmark": "local-example",
    "taskId": 0,
    "revision": 1,
    "instruction": "Read the page title and finish with it.",
    "startUrl": "https://example.com",
    "expectedAction": "RETRIEVE"
  },
  "authority": {
    "riskMode": "read_only"
  },
  "browser": {
    "headless": false,
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
