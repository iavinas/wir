// The two MID-EPISODE nudges (agent/loop.ts). Everything else the loop injects
// fires at the END — the mutate ledger, the skeptical evaluator, the give_up
// bounce, FORCE_FINISH — and the 106-episode human-driver sweep put the
// recoverable losses upstream of all of it (debug/runs/human-reddit/NOTES.md:
// 87 pass, 0 runtime-owned, 14 driver-owned losses that are two shapes —
// settling before enumerating, and concluding from a route never tried).
//
// Measured over the 708-episode trajectory corpus, which is what fixes the two
// bounds these tests pin:
//   - 0 of 12 repeated rejections in the corpus were CONSECUTIVE calls; all 12
//     were interleaved, median 2 intervening wir calls. A "same call twice in a
//     row" detector fires zero times on real episodes, so the signature spans the
//     episode and the interleaving test below is the load-bearing one.
//   - the most distinct repeating signatures any episode reached was 1 (2 at the
//     loosest signature bound), so the per-episode limit of 3 is a guard against
//     a pathological episode, never the thing that decides whether a nudge fires.
//
// The agent binary is run for real against a local OpenAI-compatible provider
// stub — no fake inside production code, no private copy of the loop.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { CONTRACT_REPLY, runAgent, toolCall } from './trajectory-rejections.test.js';

// A call core refuses on any page, with a literal repair attached: `find` with no
// filters. Page-independent on purpose — this pins the escalation, not the site.
const DOOMED_FIND = toolCall('find', '{}');
const GIVE_UP = toolCall('give_up', '{"reason":"the same call keeps being refused"}');

test('the second rejection of the same call is escalated once, and the third is clean',
  async () => {
    const { trajectory } = await runAgent({
      // 20 calls keeps the budget checkpoint (two thirds = 13) out of an episode
      // that ends at 6, so only one nudge is under test here.
      turns: [CONTRACT_REPLY, DOOMED_FIND, DOOMED_FIND, DOOMED_FIND, GIVE_UP, GIVE_UP],
      maxModelCalls: 20,
    });

    const nudges = trajectory.filter(r => r['kind'] === 'repeat_rejection_nudge');
    assert.equal(nudges.length, 1,
      `three rejections of one call earn exactly one escalation: ${JSON.stringify(nudges)}`);
    assert.equal(nudges[0]!['occurrences'], 2, 'it fires on the second, not the third');
    assert.equal(nudges[0]!['verb'], 'find');
    assert.equal(nudges[0]!['rejectionKind'], 'invalid_args');
    // The advice is the runtime's own, never invented here: every repeating
    // rejection in the corpus (28 of 28) carried a repair string.
    assert.match(String(nudges[0]!['repair']), /"verb":"find"/);

    // All three calls still reached the runtime — a nudge is not a refusal.
    const finds = trajectory.filter(
      r => r['kind'] === 'wir' && (r['request'] as { verb: string }).verb === 'find');
    assert.equal(finds.length, 3, `the escalation never blocks a call: ${finds.length}`);
  });

// The measured shape: repeats are INTERLEAVED, never consecutive. Four doomed
// calls round-robin twice, so every second rejection is 3 calls after its first.
// A detector keyed on the previous call would fire zero times here — which is
// exactly what it does on the real corpus.
test('escalation spans the episode, and stops at the per-episode limit', async () => {
  const A = toolCall('find', '{}');
  const B = toolCall('read', '{"target":"n_000000000000"}');
  const C = toolCall('act', '{"ref":"n_000000000000","action":"click"}');
  const D = toolCall('navigate', '{"url":"http://example.invalid/nowhere"}');
  const { trajectory } = await runAgent({
    turns: [CONTRACT_REPLY, A, B, C, D, A, B, C, D, GIVE_UP, GIVE_UP],
    maxModelCalls: 24,   // two thirds = 16; the episode ends at 11
  });

  const nudges = trajectory.filter(r => r['kind'] === 'repeat_rejection_nudge');
  assert.equal(nudges.length, 3,
    `four repeated calls, three escalations — the limit binds: ${JSON.stringify(
      nudges.map(n => [n['verb'], n['rejectionKind']]))}`);
  assert.deepEqual(nudges.map(n => n['occurrences']), [2, 2, 2],
    'each fires on its own second rejection, three calls after the first');
  assert.deepEqual(nudges.map(n => n['verb']), ['find', 'read', 'act'],
    'the first three repeats to come back are the three that are escalated');
  assert.deepEqual(nudges.map(n => n['nudge']), [1, 2, 3], 'and the count is on the record');
});

test('WIR_REPEAT_NUDGE=0 turns the escalation off, and says so on the episode record',
  async () => {
    const { trajectory } = await runAgent({
      turns: [CONTRACT_REPLY, DOOMED_FIND, DOOMED_FIND, DOOMED_FIND, GIVE_UP, GIVE_UP],
      maxModelCalls: 20,
      env: { WIR_REPEAT_NUDGE: '0' },
    });

    assert.equal(trajectory.filter(r => r['kind'] === 'repeat_rejection_nudge').length, 0);
    const episode = trajectory.find(r => r['kind'] === 'episode');
    assert.deepEqual(episode?.['nudges'], { repeatRejection: false, budgetCheckpoint: true },
      'which arm produced the episode must be readable off the episode');
  });

const READ = toolCall('read', '{}');

test('the budget checkpoint fires once at two thirds, carrying the contract verbatim',
  async () => {
    // The stub's turns may be functions of the literal request body — the one way
    // to assert on what the model ACTUALLY RECEIVED rather than on what the loop
    // wrote down about it.
    const bodies: string[] = [];
    const capture = (body: string): unknown => { bodies.push(body); return READ; };
    const { trajectory } = await runAgent({
      // 9 calls: two thirds is 6, the forced-finish window opens at 7.
      turns: [CONTRACT_REPLY, READ, READ, READ, READ, READ, capture, GIVE_UP, GIVE_UP],
      maxModelCalls: 9,
    });

    const checkpoints = trajectory.filter(r => r['kind'] === 'budget_checkpoint');
    assert.equal(checkpoints.length, 1,
      `once per episode: ${JSON.stringify(trajectory.map(r => r['kind']))}`);
    assert.equal(checkpoints[0]!['call'], 6, 'at two thirds of the declared budget');
    assert.equal(checkpoints[0]!['checkpointAt'], 6);

    // The next call is the one that carries it, and it carries the model's own
    // criteria, not a restatement of them.
    assert.equal(bodies.length, 1, 'the capture turn ran exactly once');
    assert.match(bodies[0]!, /6 of your 9 model calls are spent/);
    assert.match(bodies[0]!, /empty kept distinct from missing/,
      "the contract's own words travel with it");
    // And it never rides along with the forced finish, which owns the last two calls.
    assert.ok(!bodies[0]!.includes('Budget is exhausted'),
      'the checkpoint and FORCE_FINISH must never arrive in the same turn');
  });

test('a MUTATE episode has no contract, so it never gets a checkpoint', async () => {
  const { trajectory } = await runAgent({
    turns: [READ, READ, READ, READ, READ, READ, GIVE_UP, GIVE_UP],
    expectedAction: 'MUTATE',
    maxModelCalls: 9,
  });

  assert.equal(trajectory.filter(r => r['kind'] === 'completion_contract').length, 0,
    'MUTATE has no contract turn — the proof ledger is that mode\'s confrontation');
  assert.equal(trajectory.filter(r => r['kind'] === 'budget_checkpoint').length, 0,
    'and with no criteria to restate there is nothing to inject');
});

test('a finish already attempted suppresses the checkpoint, even when it was bounced',
  async () => {
    const cite = (body: string): unknown => {
      const refs = [...new Set([...body.matchAll(/n_[0-9a-f]{12,40}/g)].map(m => m[0]))];
      return toolCall('finish', JSON.stringify({
        answer: 'a', evidenceRefs: refs.slice(0, 2), status: 'success' }));
    };
    const { trajectory } = await runAgent({
      // contract(1) read(2) finish(3) -> evaluator(4) REJECTS -> bounce, then the
      // episode runs on past two thirds (8 of 12) with no second finish.
      turns: [CONTRACT_REPLY, READ, cite,
        toolCall('verdict', '{"approve":false,"objection":"the population was not read"}'),
        READ, READ, READ, READ, GIVE_UP, GIVE_UP],
      maxModelCalls: 12,
    });

    assert.equal(trajectory.filter(r => r['kind'] === 'finish_evaluation').length, 1,
      'the finish was attempted and bounced');
    assert.equal(trajectory.filter(r => r['kind'] === 'budget_checkpoint').length, 0,
      'a model that has already reached for the end is not told to check its criteria');
  });

test('WIR_BUDGET_CHECKPOINT=0 turns the checkpoint off', async () => {
  const { trajectory } = await runAgent({
    turns: [CONTRACT_REPLY, READ, READ, READ, READ, READ, READ, GIVE_UP, GIVE_UP],
    maxModelCalls: 9,
    env: { WIR_BUDGET_CHECKPOINT: '0' },
  });

  assert.equal(trajectory.filter(r => r['kind'] === 'budget_checkpoint').length, 0);
  const episode = trajectory.find(r => r['kind'] === 'episode');
  assert.deepEqual(episode?.['nudges'], { repeatRejection: true, budgetCheckpoint: false });
});
