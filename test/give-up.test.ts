// Regression for the D1 defect: `not_found_error` — added for tasks 22/24 to
// mean "the entity does not exist" — was being used to mean "I failed", because
// failure had no channel at all.
//
// Replay evidence, gitlab-10 (wir-v1 benchmark-results/local-single):
//   442  finish{answer:"not_found_error", status:"not_found_error"} — ACCEPTED
//        by the MUTATE gate (which checks acts, not status), evaluator 0. The
//        literal string was the answer.
//   659  finish{answer:"Partially completed. I successfully invi…",
//        status:not_found_error} — correctly rejected.
//   Both landed on model call 60 of 60, i.e. the FORCED FINAL call, with
//   "Budget is exhausted. Call finish now…" in the transcript (confirmed in each
//   attempt's stderr.log). The model was not confused about the world; it was
//   out of vocabulary.
//
// give_up is AGENT-LOCAL: it never reaches session.dispatch, so core's five
// verbs and the finish gate are untouched (ADR-003), and the episode is SCORED
// rather than excluded — the harness half is pinned in
// benchmark/tests/test_give_up.py.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { CONTRACT_REPLY, runAgent, toolCall } from './trajectory-rejections.test.js';

const GIVE_UP = toolCall('give_up', '{"reason":"could not find the Save control"}');

test('give_up ends the episode as a scored agent_abstained', async () => {
  const { response, trajectory } = await runAgent({
    // Two: the first is bounced once (see below), the second is the abstention.
    turns: [GIVE_UP, GIVE_UP],
    expectedAction: 'MUTATE',
    maxModelCalls: 4,
  });

  assert.equal(response['status'], 'agent_abstained', JSON.stringify(response));

  // The episode still hands the evaluator a well-formed envelope — an abstention
  // is scored, so it must not be a shape the official evaluator chokes on.
  const final = response['finalResponse'] as Record<string, unknown>;
  assert.equal(final['task_type'], 'MUTATE', JSON.stringify(final));
  assert.equal(final['status'], 'UNKNOWN_ERROR', JSON.stringify(final));
  assert.equal(final['retrieved_data'], null, JSON.stringify(final));

  // The debug plane records the reason in the model's own words.
  const gaveUp = trajectory.find(r => r['kind'] === 'give_up');
  assert.ok(gaveUp, JSON.stringify(trajectory));
  assert.equal(gaveUp['reason'], 'could not find the Save control');

  // It stopped there: nothing ran after the abstention was recorded.
  assert.equal(trajectory.filter(r => r['kind'] === 'model').length, 2,
    'give_up must end the episode, not merely be recorded');
});

// The other half of the same defect the finish confrontation was built for:
// premature termination. `finish` got a bounce because 30 of 41 failures ended
// with the agent reporting success while the evaluator scored 0; the tool whose
// ONLY function is termination got no guard at all. 8 abstentions across the
// failed-10 studies, one of them at call 54 of 60 with six calls still in hand
// (failed10-mimo-1/task-612/attempt-1).
//
// 60 links is one over the overview's 50-control page, so `read {}` withholds
// exactly 10 and mints {"verb":"read","cursor":"c_50"} to reach them — the
// mechanical fact "you never looked at this" that a give_up reason of "I could
// not find it" is answerable by.
const MANY_LINKS = `<!doctype html><title>a</title><h1>Host</h1>${
  Array.from({ length: 60 }, (_, i) => `<a href="/l${i}">link ${i}</a>`).join('')}`;

test('the first give_up is bounced with the pagination it never called back', async () => {
  const { response, trajectory } = await runAgent({
    turns: [CONTRACT_REPLY, toolCall('read', '{}'), GIVE_UP, GIVE_UP],
    maxModelCalls: 6,
    html: MANY_LINKS,
  });

  const bounces = trajectory.filter(r => r['kind'] === 'give_up_confrontation');
  assert.equal(bounces.length, 1,
    `exactly one bounce per episode: ${JSON.stringify(trajectory)}`);
  const bounce = bounces[0]!;
  // The model's own reason is kept beside the facts, verbatim.
  assert.equal(bounce['reason'], 'could not find the Save control');
  const unread = bounce['unread'] as { call: string; withheldCount: number | null }[];
  const controls = unread.find(o => o.call === '{"verb":"read","cursor":"c_50"}');
  assert.ok(controls, `the unread controls page must be reported: ${JSON.stringify(unread)}`);
  assert.equal(controls.withheldCount, 10, 'the offer carries what it reaches');

  // A bounce, not a gate: the second give_up ends the episode.
  assert.equal(response['status'], 'agent_abstained', JSON.stringify(response));
  assert.ok(trajectory.find(r => r['kind'] === 'give_up'),
    `the re-issued give_up must terminate: ${JSON.stringify(trajectory)}`);
});

// The guard that keeps the bounce from becoming a trap: an abstention is a real,
// scored outcome, and a model with no turn left to re-issue must never be
// bounced into budget_exhausted instead.
test('a give_up with no turn left to spend is never bounced', async () => {
  const { response, trajectory } = await runAgent({
    turns: [GIVE_UP],
    expectedAction: 'MUTATE',
    maxModelCalls: 1,
  });

  assert.equal(trajectory.filter(r => r['kind'] === 'give_up_confrontation').length, 0,
    'bouncing a model that cannot call again turns an honest abstention into a budget death');
  assert.equal(response['status'], 'agent_abstained', JSON.stringify(response));
});

// The ordering hazard: the WIR-budget guard refuses everything except `finish`
// once the budget is spent. If give_up were handled after it, a model that had
// run out of calls would be told "only finish is allowed now" — forced back into
// the fabricated-finish corner this whole channel exists to remove.
test('give_up still works when the WIR call budget is exhausted', async () => {
  const { response, trajectory } = await runAgent({
    turns: [toolCall('read', '{}'), toolCall('give_up', '{"reason":"out of calls"}')],
    expectedAction: 'MUTATE',
    maxModelCalls: 4,
    maxWirCalls: 1,          // spent by the first read
  });

  assert.equal(response['status'], 'agent_abstained', JSON.stringify(response));
  const gaveUp = trajectory.find(r => r['kind'] === 'give_up');
  assert.ok(gaveUp, `give_up must survive an exhausted WIR budget: ${JSON.stringify(trajectory)}`);
  assert.equal(gaveUp['reason'], 'out of calls');
  const refused = trajectory.filter(r => r['kind'] === 'rejected' && r['name'] === 'give_up');
  assert.equal(refused.length, 0, 'give_up must never be refused for budget');
});

test('give_up never reaches the runtime — the finish gate keeps its authority', async () => {
  const blocked = toolCall('give_up', '{"reason":"blocked"}');
  const { trajectory } = await runAgent({
    turns: [blocked, blocked],
    expectedAction: 'MUTATE',
    maxModelCalls: 4,
  });

  // No wir record for it: core's dispatch never saw a sixth verb.
  const wirVerbs = trajectory
    .filter(r => r['kind'] === 'wir')
    .map(r => (r['request'] as { verb: string }).verb);
  assert.ok(!wirVerbs.includes('give_up'),
    `give_up must stay agent-local: ${JSON.stringify(wirVerbs)}`);
});
