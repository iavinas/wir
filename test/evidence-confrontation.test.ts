// Regression for the D2 defect (spec step 20): the MUTATE gate proves that *a*
// mutation happened, never that the whole task did.
//
// Replay evidence, gitlab-10 (wir-v1 benchmark-results/local-single):
//   743 "Create a new public project web_arena and add Abishek, Vinta as
//       members" — 19 acts, finish cited 15 refs, gate ACCEPTED, evaluator 0:
//       AgentResponseEvaluator 1.0, all three NetworkEventEvaluator criteria 0.
//   747 "Start a private project … and add Abishek, Vinta as members" — finish
//       rejected twice (cited a_6 unknown + a_18 dom_mutated, both ineligible),
//       accepted on the third try; evaluator 0 overall, though the members
//       criterion scored 1.0 and the project-creation POST did not match.
// A multi-criterion task can satisfy the gate on step 1 and fail on step 3, and
// nothing in the runtime can know the task had three steps.
//
// The bounce is pure information-surfacing: the ledger is facts the model
// already had, collected at the moment it matters. No comparison of the answer
// to the instruction happens anywhere (ADR-003 untouched).
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { CONTRACT_REPLY, runAgent, textReply, toolCall } from './trajectory-rejections.test.js';

const FINISH = toolCall('finish', JSON.stringify({
  answer: 'done', evidenceRefs: ['a_1'], status: 'success',
}));

test('a MUTATE episode bounces the first finish with the proof ledger', async () => {
  const { trajectory } = await runAgent({
    turns: [FINISH, FINISH, toolCall('read', '{}')],
    expectedAction: 'MUTATE',
    maxModelCalls: 5,
  });

  const bounce = trajectory.find(r => r['kind'] === 'evidence_confrontation');
  assert.ok(bounce, `the first MUTATE finish must be confronted: ${JSON.stringify(trajectory)}`);
  // The model's own finish arguments are kept verbatim beside the ledger.
  assert.match(String(bounce['arguments']), /"answer":"done"/);
  // Nothing was proved in this episode, and the ledger says so rather than
  // staying silent.
  assert.deepEqual(bounce['ledger'], []);

  // Exactly once per episode: the second finish is not bounced again.
  assert.equal(trajectory.filter(r => r['kind'] === 'evidence_confrontation').length, 1);
  const finishes = trajectory.filter(
    r => r['kind'] === 'wir' && (r['request'] as { verb: string }).verb === 'finish');
  assert.equal(finishes.length, 1,
    `the second finish must reach the runtime: ${JSON.stringify(finishes)}`);
});

// The RETRIEVE/NAVIGATE half. Rebuilt twice, each time on a measurement. Bytes
// -> scope (2026-08-11): the byte-ranked bounce fired on 98% of episodes and
// changed the answer in ~2% (n=57). Scope-trigger -> evaluator (2026-08-13):
// the scope trigger fired only when a cited population was PARTLY delivered,
// and 8 of 10 agent losses were scope decisions made with COMPLETE data in
// hand — the trigger structurally could not see the loss class it was built
// for. Now EVERY first RETRIEVE/NAVIGATE finish is judged by one extra model
// call on a fresh conversation carrying only runtime-known facts, whose stance
// is refutation; a rejection bounces once with the objection, an approval (or
// an unparseable verdict, failing open) lets the finish through to the gate.
// The gate itself is untouched: the objection is advice to the model, never
// authority (ADR-003).
//
// The byte ledger is NOT gone: give_up still uses it, where "content you never
// read" is exactly the right thing to say to a model about to abandon.
//
// 60 links is one over the overview's 50-control page, so `read {}` withholds
// exactly 10 and mints {"verb":"read","cursor":"c_50"} to reach them — the
// give_up test below still leans on that.
const MANY_LINKS = `<!doctype html><title>a</title><h1>Host</h1>${
  Array.from({ length: 60 }, (_, i) => `<a href="/l${i}">link ${i}</a>`).join('')}`;

// A list longer than one page of items: the overview delivers 10 of 24, so any ref
// inside it names a population this episode has seen only part of.
const LONG_LIST = `<!doctype html><title>a</title><h1>Host</h1><ul>${
  Array.from({ length: 24 }, (_, i) => `<li>row ${i}</li>`).join('')}</ul>`;

// Cite the rows the overview actually delivered, read out of the stub's own
// transcript the way the model reads them — refs are runtime-minted and no
// fixed script can know them.
const citeDelivered = (body: string) => {
  const refs = [...new Set([...body.matchAll(/n_[0-9a-f]{12,40}/g)].map(m => m[0]))];
  return toolCall('finish', JSON.stringify({
    answer: 'row 0', evidenceRefs: refs, status: 'success' }));
};

test('a rejected evaluation bounces once with the objection, and only once',
  async () => {
    const OBJECTION =
      'Only part of the population was read; members never seen could not be ruled out.';
    const { response, trajectory } = await runAgent({
      turns: [CONTRACT_REPLY, toolCall('read', '{}'), citeDelivered,
        // The evaluator conversation consumes the next scripted turn: its
        // verdict arrives through the forced verdict tool (sweep17 measured
        // the JSON-in-text transport broken on mimo, 4 of 7 unparseable).
        toolCall('verdict', JSON.stringify({ approve: false, objection: OBJECTION })),
        citeDelivered, toolCall('read', '{}')],
      expectedAction: 'RETRIEVE',
      maxModelCalls: 8,
      html: LONG_LIST,
    });

    const evals = trajectory.filter(r => r['kind'] === 'finish_evaluation');
    assert.equal(evals.length, 1,
      `one evaluation per episode — the bounce consumes the flag: ${JSON.stringify(evals)}`);
    assert.equal(evals[0]!['verdict'], 'rejected');
    assert.equal(evals[0]!['objection'], OBJECTION,
      'the objection travels to the model verbatim');
    const facts = evals[0]!['facts'] as { items: number; delivered: number }[];
    assert.ok(Array.isArray(facts) && facts.length > 0,
      `the evaluator saw the runtime's scope facts: ${JSON.stringify(evals[0])}`);

    // The bounce costs one turn and no more: the re-submitted finish reaches
    // WIR without a second evaluation, and the episode succeeds.
    const finishes = trajectory.filter(
      r => r['kind'] === 'wir' && (r['request'] as { verb: string }).verb === 'finish');
    assert.equal(finishes.length, 1,
      `the second finish must reach the runtime: ${JSON.stringify(finishes)}`);
    assert.equal(response['status'], 'success', JSON.stringify(response));
  });

test('an approved evaluation lets the first finish through to the gate', async () => {
  const { response, trajectory } = await runAgent({
    turns: [CONTRACT_REPLY, toolCall('read', '{}'), citeDelivered,
      toolCall('verdict', '{"approve": true}'), toolCall('read', '{}')],
    expectedAction: 'RETRIEVE',
    maxModelCalls: 6,
    html: MANY_LINKS,
  });

  const evals = trajectory.filter(r => r['kind'] === 'finish_evaluation');
  assert.equal(evals.length, 1, `every first finish is evaluated: ${JSON.stringify(trajectory)}`);
  assert.equal(evals[0]!['verdict'], 'approved');
  assert.equal(trajectory.filter(r => r['kind'] === 'evidence_confrontation').length, 0,
    'an approval is not a bounce — the model never hears from the evaluator');
  const finishes = trajectory.filter(
    r => r['kind'] === 'wir' && (r['request'] as { verb: string }).verb === 'finish');
  assert.equal(finishes.length, 1, 'and the first finish reaches the runtime directly');
  assert.equal(response['status'], 'success', JSON.stringify(response));
});

// CHANGED 2026-08-20, deliberately. This pinned "unparseable == approved", and
// that is the defect: a guard that ran and produced garbage counted as a guard
// that passed. Measured on human-57/task-57/attempt-12 — verdict "unparseable",
// answer 6 of 7 restaurants, accepted — and the surveyed prior art is unanimous
// that guards fail CLOSED (NeMo Guardrails, VLAA-GUI, LangGraph interrupts).
//
// The original concern survives and is why this is BOUNDED: a provider that can
// never call the tool (the recorded mimo prose shape, sweep17-integration/
// task-235/attempt-1 record 15) must not be bounced until its budget is gone. So
// the FIRST unparseable verdict costs one retryable finish and the SECOND fails
// open. Both halves are pinned below.
test('an unparseable verdict bounces once, then fails open',
  async () => {
    const { response, trajectory } = await runAgent({
      // Two finishes, each followed by a prose (unparseable) evaluator reply.
      turns: [CONTRACT_REPLY, toolCall('read', '{}'), citeDelivered,
        textReply('Looking at the runtime observations, the answer seems incomplete.'),
        citeDelivered,
        textReply('Still prose, still not a verdict.'),
        toolCall('read', '{}')],
      expectedAction: 'RETRIEVE',
      maxModelCalls: 9,
      html: MANY_LINKS,
    });

    const evals = trajectory.filter(r => r['kind'] === 'finish_evaluation');
    assert.ok(evals.length >= 1, `finishes are evaluated: ${JSON.stringify(trajectory)}`);
    assert.equal(evals[0]!['verdict'], 'unparseable',
      `prose is not a verdict: ${JSON.stringify(evals[0])}`);

    // THE CHANGE: the first unparseable verdict is not an approval.
    assert.equal(
      trajectory.filter(r => r['kind'] === 'verdict_unparseable_bounce').length, 1,
      `an unreadable verdict must bounce exactly once: ${JSON.stringify(trajectory)}`);

    // AND THE BOUND: it does not brick the episode — the second one lets it through.
    const finishes = trajectory.filter(
      r => r['kind'] === 'wir' && (r['request'] as { verb: string }).verb === 'finish');
    assert.ok(finishes.length >= 1,
      `a guard must never brick an episode: ${JSON.stringify(trajectory)}`);
    assert.equal(response['status'], 'success', JSON.stringify(response));
  });

test('give_up still reports unread pagination, and not one the model took', async () => {
  // The byte ledger keeps its one legitimate use. To a model about to abandon,
  // "content the runtime holds and you never read" is exactly the right thing to
  // say — a thing you never looked at is not a thing that is not there.
  const { trajectory } = await runAgent({
    turns: [CONTRACT_REPLY, toolCall('read', '{}'), toolCall('read', '{"cursor":"c_50"}'),
      toolCall('give_up', JSON.stringify({ reason: 'stuck' })),
      toolCall('give_up', JSON.stringify({ reason: 'stuck' })), toolCall('read', '{}')],
    expectedAction: 'RETRIEVE',
    maxModelCalls: 7,
    html: MANY_LINKS,
  });

  const bounce = trajectory.find(r => r['kind'] === 'give_up_confrontation');
  assert.ok(bounce, `give_up is confronted once: ${JSON.stringify(trajectory)}`);
  const unread = bounce['unread'] as { call: string }[];
  assert.ok(!unread.some(o => o.call.includes('c_50')),
    `an offer that was taken is not an unread offer: ${JSON.stringify(unread)}`);
});

// The guard that keeps the bounce from costing an episode its finish: a model on
// its forced final call has no turn left to spend on a re-submit.
test('the forced final call is never bounced', async () => {
  const { trajectory } = await runAgent({
    turns: [toolCall('read', '{}'), FINISH],
    expectedAction: 'MUTATE',
    maxModelCalls: 2,
  });

  assert.equal(trajectory.filter(r => r['kind'] === 'evidence_confrontation').length, 0,
    'bouncing a model that cannot call again would destroy a finish it had earned');
  const finishes = trajectory.filter(
    r => r['kind'] === 'wir' && (r['request'] as { verb: string }).verb === 'finish');
  assert.equal(finishes.length, 1, 'the final finish must reach the runtime');
});
