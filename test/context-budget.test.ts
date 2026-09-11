// Regression for review finding R9: the agent accumulated inputTokens and never
// compared them to anything. A context overflow therefore surfaced as a
// non-retryable provider 4xx -> ProviderError -> provider_error, which the
// runner EXCLUDES — the agent's own failure laundered out of the denominator,
// inflating the pass rate by shrinking it. Recorded episodes already reach
// ~110-115K-token final prompts, so this is a live boundary.
//
// The end-to-end path is exercised through the real agent binary and the real
// stdin/stdout protocol; the runner's half (a scored budget_exhausted) is pinned
// in benchmark/tests/test_context_budget.py.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { CONTRACT_REPLY, runAgent, toolCall } from './trajectory-rejections.test.js';

// The stub reports 1000 input tokens per call (see providerStub), so a budget of
// 500 is breached by the FIRST call's usage — here the completion contract
// (RETRIEVE, 6 calls) — and the next call must be the forced final one.
test('breaching the context budget ends the episode as budget_exhausted', async () => {
  const { response, trajectory } = await runAgent({
    turns: [CONTRACT_REPLY, toolCall('read', '{}'), toolCall('read', '{}')],
    maxModelCalls: 6,
    maxInputTokensPerCall: 500,
  });

  assert.equal(response['status'], 'budget_exhausted', JSON.stringify(response));
  // Exactly one call past the breach: the FORCE_FINISH exemption must fire once
  // — without it the agent could never finish at all — and never twice. The
  // contract call observed the breach, so a single loop turn follows it.
  assert.equal(trajectory.filter(r => r['kind'] === 'completion_contract').length, 1,
    `the contract call observes the breach: ${JSON.stringify(trajectory)}`);
  const modelCalls = trajectory.filter(r => r['kind'] === 'model');
  assert.equal(modelCalls.length, 1,
    `one forced final after the breach: ${JSON.stringify(modelCalls)}`);
  const end = trajectory.find(r => r['kind'] === 'end');
  assert.equal(end?.['status'], 'budget_exhausted', JSON.stringify(end));
});

test('an undeclared context budget is unlimited', async () => {
  const { response, trajectory } = await runAgent({
    turns: [toolCall('read', '{}'), toolCall('read', '{}'), toolCall('read', '{}')],
    maxModelCalls: 3,
  });

  // No early stop: the episode runs its full model-call budget as before the
  // field existed.
  assert.equal(response['status'], 'budget_exhausted', JSON.stringify(response));
  assert.equal(trajectory.filter(r => r['kind'] === 'model').length, 3,
    'without a declared budget the loop must not stop on token count');
});
