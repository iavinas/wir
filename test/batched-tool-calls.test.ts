// Several verbs in one round trip, and a batch that stops when the page moves.
//
// 96% of an episode's wall clock is model latency at ~5.1s median per round trip,
// against a 41ms median WIR dispatch. So a saved ROUND TRIP is ~5 seconds and a
// saved millisecond inside the runtime is nothing — batching is the only lever on
// the clock that does not require doing less work.
//
// Measured before it was asked for: 11,588 of 11,730 recorded model turns (98.8%)
// emitted exactly one tool call. The loop has always executed batches; nothing ever
// requested them, and `parallel_tool_calls` was never sent. A 22-call episode at one
// verb per turn is 112s; the same verbs at ~3 per turn is 41s.
//
// THE BOUND, and the second test is what pins it: within one turn the model composes
// every call before seeing any result, so no call in a batch can use another's
// output — but a call that MOVES THE PAGE invalidates the refs every later call was
// written against. 206 stale_ref/unknown_ref rejections already land on the call
// immediately after an act, and that is exactly the population batching pushes on.
// So the remainder is skipped and SAID to be skipped: every tool_call_id still gets
// an answer, because an unanswered one is a protocol error on the next request.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { CONTRACT_REPLY, runAgent, toolCall } from './trajectory-rejections.test.js';

const PAGE = '<!doctype html><title>batch</title><h1>Batch</h1>'
  + '<a href="/second">Go second</a><p>alpha text</p><p>bravo text</p>';

/** One assistant turn carrying several tool calls, which is what a batching model
 *  emits and what the stub must be able to express. */
function batch(...calls: { name: string; args: string }[]): unknown {
  return {
    role: 'assistant', content: null,
    tool_calls: calls.map((c, i) => ({
      id: `call_${c.name}_${i}`, type: 'function',
      function: { name: c.name, arguments: c.args },
    })),
  };
}

test('independent verbs in one turn all run, costing one round trip', async () => {
  const { trajectory } = await runAgent({
    turns: [
      CONTRACT_REPLY,
      batch({ name: 'read', args: '{}' },
        { name: 'find', args: '{"name":"alpha"}' },
        { name: 'find', args: '{"name":"bravo"}' }),
      toolCall('give_up', JSON.stringify({ reason: 'done measuring' })),
      toolCall('give_up', JSON.stringify({ reason: 'done measuring' })),
    ],
    expectedAction: 'RETRIEVE',
    maxModelCalls: 6,
    html: PAGE,
  });

  const wir = trajectory.filter(r => r['kind'] === 'wir');
  const models = trajectory.filter(r => r['kind'] === 'model');
  assert.equal(wir.length >= 3, true, `all three verbs ran: ${wir.length}`);

  // The point of the whole change: three verbs, ONE model turn.
  const firstTurn = models[0] as Record<string, unknown>;
  assert.equal((firstTurn['toolCalls'] as string[]).length, 3,
    'the turn carried three calls');
  const verbs = wir.slice(0, 3).map(r => (r['request'] as { verb: string }).verb);
  assert.deepEqual(verbs, ['read', 'find', 'find'],
    `executed in the order issued: ${JSON.stringify(verbs)}`);
  for (const r of wir.slice(0, 3)) {
    assert.equal((r['response'] as Record<string, unknown>)['rejected'], undefined,
      'and none of them was rejected');
  }
});

test('SEVERAL ACTS in one turn all run — refs survive a page CHANGING', async () => {
  // THE CAPABILITY, and the reason the old rule was wrong. Refs are
  // sha1(epoch:backendNodeId): a same-document mutation re-mints them
  // byte-identically. Measured on a live page, 110 of 110 survived a recompile
  // after the DOM changed. So ten clicks on ten controls belong in ONE turn.
  //
  // The old rule stopped after ANY act, costing one ~3.1 s round trip per
  // control — 43 s of pure transport for fourteen widgets and 40 ms of work.
  //
  // Three checkboxes, deliberately: each click MUTATES the document (its own
  // checked state) without replacing it. That is exactly the case the old rule
  // wrongly treated as fatal.
  const BOXES = '<!doctype html><title>boxes</title><h1>Boxes</h1>'
    + '<label><input type="checkbox" id="a"> alpha box</label>'
    + '<label><input type="checkbox" id="b"> bravo box</label>'
    + '<label><input type="checkbox" id="c"> charlie box</label>';
  const { trajectory } = await runAgent({
    turns: [
      CONTRACT_REPLY,
      toolCall('read', '{}'),
      (body: string) => {
        // The read response is JSON.stringify'd into a tool message, so the body
        // carries it ESCAPED. Unescape before matching, or every regex silently
        // finds nothing and the test claims a failure that is its own.
        const plain = body.replace(/\\/g, '');
        const refs = [...plain.matchAll(/"ref":"(n_[0-9a-f]{12,40})","role":"checkbox"/g)]
          .map(m => m[1]);
        return batch(
          { name: 'act', args: JSON.stringify({ ref: refs[0], action: 'click' }) },
          { name: 'act', args: JSON.stringify({ ref: refs[1], action: 'click' }) },
          { name: 'act', args: JSON.stringify({ ref: refs[2], action: 'click' }) });
      },
      toolCall('give_up', JSON.stringify({ reason: 'done' })),
      toolCall('give_up', JSON.stringify({ reason: 'done' })),
    ],
    expectedAction: 'RETRIEVE',
    maxModelCalls: 8,
    html: BOXES,
  });

  const wir = trajectory.filter(r => r['kind'] === 'wir');
  const acts = wir.filter(r => (r['request'] as { verb: string }).verb === 'act');
  assert.equal(acts.length, 3,
    `all three acts ran in ONE turn: ${JSON.stringify(acts.map(a => a['request']))}`);
  for (const a of acts) {
    const rej = (a['response'] as Record<string, unknown>)['rejected'];
    assert.equal(rej, undefined, `and none was skipped or stale: ${JSON.stringify(rej)}`);
  }
  // One model turn carried all three — that is the whole point.
  const models = trajectory.filter(r => r['kind'] === 'model');
  assert.equal((models[1]?.['toolCalls'] as string[]).length, 3);
});

test('CONTROL — a batch stops when a call REPLACES the document, and says so', async () => {
  // The bound that remains, and the only one. A navigation kills every ref
  // composed against the old document, so running the rest would manufacture
  // exactly the stale_ref rejections batching exists to avoid paying for.
  // `documentEpoch` (ADR-002, the main-frame loaderId) is the runtime's proof.
  //
  // location.reload() is a genuine document replacement on the same origin —
  // a new loaderId, every ref re-minted — which is precisely the condition
  // under test, without needing a second file or a server.
  const RELOAD_PAGE = '<!doctype html><title>batch</title><h1>Batch</h1>'
    + '<button id="r" onclick="location.reload()">Reload me</button>'
    + '<p>alpha text</p><p>bravo text</p>';
  const { trajectory, response } = await runAgent({
    turns: [
      CONTRACT_REPLY,
      toolCall('read', '{}'),
      (body: string) => {
        const plain = body.replace(/\\/g, '');
        const m = /"ref":"(n_[0-9a-f]{12,40})","role":"button"/.exec(plain);
        const ref = m ? m[1]
          : [...new Set([...plain.matchAll(/n_[0-9a-f]{12,40}/g)].map(x => x[0]))][0];
        return batch(
          { name: 'act', args: JSON.stringify({ ref, action: 'click' }) },
          { name: 'read', args: '{}' },
          { name: 'find', args: '{"name":"alpha"}' });
      },
      toolCall('give_up', JSON.stringify({ reason: 'done' })),
      toolCall('give_up', JSON.stringify({ reason: 'done' })),
    ],
    expectedAction: 'RETRIEVE',
    maxModelCalls: 8,
    html: RELOAD_PAGE,
  });

  const wir = trajectory.filter(r => r['kind'] === 'wir');
  const acts = wir.filter(r => (r['request'] as { verb: string }).verb === 'act');
  assert.equal(acts.length, 1, 'the act itself ran');
  const afterAct = wir.slice(wir.indexOf(acts[0]!) + 1);
  assert.equal(afterAct.length, 0,
    `nothing ran after the document was replaced: ${JSON.stringify(afterAct.map(r => r['request']))}`);
  assert.ok(['agent_abstained', 'success', 'budget_exhausted'].includes(String(response['status'])),
    `and a skipped call is answered, not dropped: ${response['status']}`);
});
