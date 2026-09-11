// Regression: a forced finish the gate REJECTED used to be unrecoverable by
// construction.
//
// `mustFinish` fired at modelCalls === maxModelCalls - 1, so FORCE_FINISH bought
// the model exactly one turn. The finish gate teaches through a repair string —
// "cite an act whose evidence proves a server-side change", "cite refs you
// received from find/read this episode" — and on the one call where finishing
// was mandatory there was no turn left to act on it.
//
// Measured, failed-10 studies (53 episodes replayed from the recorded
// trajectories): 29 gate rejections, and three of them landed on the final call
// with nothing after them —
//   benchmark-results/failed10-current-1/task-442/attempt-1  (call 60 of 60)
//   benchmark-results/failed10-current-1/task-659/attempt-1  (call 60 of 60)
//   benchmark-results/failed10-proj/task-442/attempt-1       (call 60 of 60)
// each ending `budget_exhausted` one call after being told exactly what to fix.
//
// The fix reserves the last call: the forced window is the last TWO, so a
// rejected forced finish always has a turn to repair. The irony this pins is
// that the confrontation guard three lines below already reasoned about this
// hazard for the bounce, and the reasoning was never applied to the gate.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { runAgent, toolCall } from './trajectory-rejections.test.js';

// Rejected by the gate on the answer, before the evidence rule is reached.
const EMPTY_FINISH = toolCall('finish', JSON.stringify({
  answer: '', evidenceRefs: [], status: 'success',
}));

// The repair: a non-empty answer citing a ref the runtime actually minted, which
// the stub lifts out of the transcript the way the model does.
function repairFinish(body: string): unknown {
  const ref = /n_[0-9a-f]{6,}/.exec(body)?.[0] ?? 'n_never_minted';
  return toolCall('finish', JSON.stringify({
    answer: '42', evidenceRefs: [ref], status: 'success',
  }));
}

test('a rejected forced finish still has a turn to repair itself', async () => {
  // Three calls: one read (so a ref exists to cite), then the forced window —
  // the finish the gate rejects, and the repair.
  const { response, trajectory } = await runAgent({
    turns: [toolCall('read', '{}'), EMPTY_FINISH, repairFinish],
    maxModelCalls: 3,
  });

  const finishes = trajectory.filter(
    r => r['kind'] === 'wir' && (r['request'] as { verb: string }).verb === 'finish');
  assert.equal(finishes.length, 2,
    `the rejected finish must be followed by a second one that reaches the gate: ${
      JSON.stringify(trajectory.filter(r => r['kind'] !== 'wir'))}`);

  const rejected = (finishes[0]!['response'] as Record<string, unknown>);
  assert.notEqual(rejected['accepted'], true, JSON.stringify(rejected));
  assert.match(String((rejected['rejected'] as Record<string, unknown>)['reason']),
    /empty answer/);
  assert.equal((finishes[1]!['response'] as Record<string, unknown>)['accepted'], true,
    'the repaired finish must be accepted');

  // The repair is not merely recorded — it is the episode's outcome.
  assert.equal(response['status'], 'success', JSON.stringify(response));
  const final = response['finalResponse'] as Record<string, unknown>;
  assert.equal(final['retrieved_data'], '42', JSON.stringify(final));

  // The forced window is never bounced: the repair turn must not be spent on a
  // confrontation.
  assert.equal(trajectory.filter(r => r['kind'] === 'evidence_confrontation').length, 0,
    'a bounce inside the forced window would consume the turn the fix reserved');
});
