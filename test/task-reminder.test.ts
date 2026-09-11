// The task, restated at the end of every observation.
//
// Delivered once in message 2 and then buried: the largest failure family in the
// corpus is "did the right kind of work, committed to the wrong thing" — of 387
// recorded failures, 223 carry a `missing_navigation_event` assertion and 170
// carry it as the only one, and three map losses were choosing the wrong instance
// of an ambiguous name.
//
// Corrected 2026-08-20, same change as agent/loop.ts:1391. This said "117 of 120
// REQUEST_MISMATCH failures"; the figures do not reproduce and REQUEST_MISMATCH is
// not a code-defined outcome — it lived only in these two comments.
//
// This pins the delivery, not the effect. The effect is a PRE-REGISTERED
// experiment recorded in debug/LEARNINGS.md and predicted to fail: advisory
// signals have not moved this model before. A test that a field arrives says
// nothing about whether it helps — that is what the controlled arm is for.
//
// Pattern copied from finish-navigate-standing.test.ts's 'every envelope carries
// the current url', the established pin for an additive field.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { runAgent, toolCall, CONTRACT_REPLY } from './trajectory-rejections.test.js';

test('the task reminder rides every observation the model receives', async () => {
  const instruction = 'Open the order details page for the most recent processing order';
  const run = await runAgent({
    instruction,
    // The harness defaults to maxModelCalls: 2 — the contract turn takes one and
    // the forced-finish window opens immediately, so no verb ever runs and the
    // payloads carry no tool results at all.
    maxModelCalls: 8,
    turns: [
      CONTRACT_REPLY,
      toolCall('read', '{}'),
      toolCall('find', '{"role":"link"}'),
      toolCall('give_up', '{"reason":"end the fixture"}'),
      toolCall('give_up', '{"reason":"end the fixture"}'),
    ],
    html: '<!doctype html><title>t</title><h1>Orders</h1><a href="#x">Order 1</a>',
  });

  // Read what the model was ACTUALLY handed, not what the trajectory recorded —
  // the reminder is appended after the trajectory entry is written.
  const msgs = run.payloads.flatMap(p => (p['messages'] as Record<string, unknown>[] | undefined) ?? []);
  const toolResults = msgs.filter(m => m['role'] === 'tool');
  assert.ok(toolResults.length >= 2,
    `expected several tool results in the payloads, got ${toolResults.length}`);

  let checked = 0;
  for (const m of toolResults) {
    const body = String(m['content'] ?? '');
    // give_up is agent-local: it never reaches session.dispatch, so its result
    // does not travel the WIR payload path and carries no reminder.
    if (body.includes('"gaveUp"') || body.includes('give_up')) continue;
    checked += 1;
    assert.ok(body.includes('taskReminder'),
      `a WIR observation reached the model without the reminder: ${body.slice(0, 220)}`);
    assert.ok(body.includes(instruction),
      `the reminder must carry the task VERBATIM, not a summary: ${body.slice(0, 220)}`);
  }
  assert.ok(checked >= 2, `expected to check at least two WIR observations, checked ${checked}`);
});
