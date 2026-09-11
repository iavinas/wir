// Filling a form cost one round trip PER FIELD, and the round trip is the bill.
//
// The loop stopped a batch at the first `act` of any kind. A six-field form was
// therefore six model turns before the submit — at ~5s of provider latency each,
// thirty seconds of waiting for work the runtime does in ~0.4s per field. The
// prompt asked the model to batch while the loop refused to let it: measured
// across 23 recorded episodes, 463 model turns carried 577 tool calls and 401 of
// those turns carried exactly one.
//
// A fill sets its own control's value and strands nothing — the refs in the rest
// of the batch still name the same nodes. Only an act whose effect reaches beyond
// its target can invalidate them, and those still end the turn (the control in
// batched-tool-calls.test.ts).
//
// The safety net is unchanged: every act revalidates its ref against the
// browser's own computation at dispatch, so a ref that HAS gone stale is refused
// with `stale_ref` rather than acted on blindly.
//
// FIXTURE JUSTIFIED: needs several independent fields in one document so that
// "they all ran, in one turn" is decidable.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { runAgent, toolCall } from './trajectory-rejections.test.js';

const PAGE = '<!doctype html><title>fills</title><h1>Fills</h1>'
  + '<form>'
  + '<label>First <input id="a" name="first"></label>'
  + '<label>Second <input id="b" name="second"></label>'
  + '<label>Third <input id="c" name="third"></label>'
  + '</form>';

function batch(...calls: { name: string; args: string }[]): unknown {
  return {
    role: 'assistant', content: null,
    tool_calls: calls.map((c, i) => ({
      id: `call_${c.name}_${i}`, type: 'function',
      function: { name: c.name, arguments: c.args },
    })),
  };
}

test('several fills in one turn all run, costing one round trip', async () => {
  const { trajectory } = await runAgent({
    turns: [
      toolCall('read', '{}'),
      (body: string) => {
        // Every textbox ref the transcript carries, in order.
        const refs = [...new Set([...body.matchAll(/(n_[0-9a-f]{12,40})\\?",\\?"role\\?":\\?"textbox/g)]
          .map(m => m[1]!))];
        return batch(...refs.slice(0, 3).map((ref, i) => ({
          name: 'act',
          args: JSON.stringify({ ref, action: 'fill', value: `value${i}` }),
        })));
      },
      toolCall('give_up', '{"reason":"done"}'),
      toolCall('give_up', '{"reason":"done"}'),
    ],
    expectedAction: 'MUTATE',
    maxModelCalls: 8,
    html: PAGE,
  });

  const wir = trajectory.filter(r => r['kind'] === 'wir');
  const fills = wir.filter(r => {
    const req = r['request'] as { verb?: string; action?: string };
    return req.verb === 'act' && req.action === 'fill';
  });
  assert.equal(fills.length, 3,
    `all three fills must run in the one turn that issued them: ${JSON.stringify(
      wir.map(r => r['request']))}`);
  for (const f of fills) {
    const res = f['response'] as Record<string, unknown>;
    assert.equal(res['rejected'], undefined,
      `a fill must not be skipped as "the page moved" — nothing moved: ${JSON.stringify(res)}`);
    const effect = res['effect'] as Record<string, unknown>;
    assert.equal(effect['evidence'], 'value_set', JSON.stringify(res));
  }

  // And it really was ONE turn, which is the entire point.
  const models = trajectory.filter(r => r['kind'] === 'model');
  const batched = models.find(m => ((m['toolCalls'] ?? []) as string[]).length === 3);
  assert.ok(batched, `one turn carried all three calls: ${JSON.stringify(
    models.map(m => m['toolCalls']))}`);
});
