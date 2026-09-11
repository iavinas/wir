// Regression for the credential-posture step: a value filled into a page can be a
// password, and telemetry ships to a third party. Redaction is STRUCTURAL — it
// fires on the position a fill value occupies in a known shape, never on a
// pattern that "looks like" a secret — and it happens at the vendor-neutral
// telemetry seam, so a second backend cannot forget to apply it.
//
// Partial redaction is worse than none, because it reads as protection. This pins
// all SIX positions a fill value reaches the boundary in. Positions 5 and 6 were
// each missed by the original design and found by adversarial review — twice, which
// is why the count now lives in a test rather than only in a comment.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { isolate, type EpisodeTrace, type GenerationRecord, type Telemetry, type ToolCallRecord }
  from '../agent/telemetry/index.js';

const SECRET = 'hunter2-correct-horse';
const MARK = `[redacted:len=${SECRET.length}]`;

function capture(): { telemetry: Telemetry; gens: GenerationRecord[]; tools: ToolCallRecord[] } {
  const gens: GenerationRecord[] = [];
  const tools: ToolCallRecord[] = [];
  const trace: EpisodeTrace = {
    traceId: 'test',
    generation(r) { gens.push(r); },
    toolCall(r) { tools.push(r); },
    end() { /* no-op */ },
  };
  const inner: Telemetry = {
    enabled: true,
    run: async (_meta, fn) => fn(trace),
    shutdown: async () => undefined,
  };
  return { telemetry: isolate(inner), gens, tools };
}

const meta = { runId: 'r', taskId: 1, instruction: 'i', expectedAction: 'MUTATE', model: 'm' };

// The shapes the agent actually emits: act.ts writes the value into the delta as
// `value="…"` and tags it with evidence value_set.
const fillArgs = { verb: 'act', ref: 'n_abc123abc123', action: 'fill', value: SECRET };
const fillResult = {
  actRef: 'a_1', outcome: 'delivered',
  effect: { verdict: 'verified', evidence: 'value_set',
    delta: { before: 'value=null', after: `value=${JSON.stringify(SECRET)}` } },
};
const assistantMessage = {
  role: 'assistant', content: null,
  tool_calls: [{ id: 'c1', type: 'function',
    function: { name: 'act', arguments: JSON.stringify(fillArgs) } }],
};

// Position 5, found by adversarial review: read.ts projects a node's CURRENT value
// on controls and children, so a filled password is re-emitted in every later read
// of that page — and in every tool-role message carrying that read into the next
// prompt. Covering four of five would have read as protection while leaking.
test('a filled value is redacted in every later read result', async () => {
  const { telemetry, gens, tools } = capture();
  const readResult = {
    documentEpoch: 'e1', freshness: 'live', coverageIncomplete: false,
    title: 'Sign in', url: 'https://example.test/login',
    controls: [
      { ref: 'n_aaaaaaaaaaaa', role: 'textbox', name: 'Password', value: SECRET },
      { ref: 'n_bbbbbbbbbbbb', role: 'button', name: 'Sign in' },
    ],
    controlsTotal: 2,
  };
  await telemetry.run(meta, async (t) => {
    t.toolCall({ verb: 'read', args: { verb: 'read' }, result: readResult, ms: 3 });
    t.generation({
      model: 'm', ms: 4, usage: { input: 1, output: 1 },
      input: { model: 'm', messages: [
        { role: 'tool', tool_call_id: 'c9', content: JSON.stringify(readResult) },
      ] },
      output: { role: 'assistant', content: null },
    });
  });

  const everything = JSON.stringify({ tools, gens });
  assert.ok(!everything.includes(SECRET),
    `a filled value must not survive into a later read: ${everything}`);
  const control = (tools[0]!.result as { controls: { value?: string }[] }).controls[0];
  assert.equal(control!.value, MARK);
  // the button, which has no value, is untouched
  assert.deepEqual((tools[0]!.result as { controls: unknown[] }).controls[1],
    { ref: 'n_bbbbbbbbbbbb', role: 'button', name: 'Sign in' });
});

test('every position a fill value reaches telemetry is redacted', async () => {
  const { telemetry, gens, tools } = capture();
  await telemetry.run(meta, async (t) => {
    // 1 + 2: the act tool span's input and its effect delta.
    t.toolCall({ verb: 'act', args: fillArgs, result: fillResult, ms: 5 });
    // 3 + 4: the next generation's transcript (assistant tool_calls and the
    // tool-role result), and the assistant output that originates the value.
    t.generation({
      model: 'm', ms: 10, usage: { input: 1, output: 1 },
      input: { model: 'm', messages: [
        assistantMessage,
        { role: 'tool', tool_call_id: 'c1', content: JSON.stringify(fillResult) },
      ] },
      output: assistantMessage,
    });
  });

  const everything = JSON.stringify({ tools, gens });
  assert.ok(!everything.includes(SECRET),
    `the value must not appear anywhere at the boundary: ${everything}`);
  // and the marker must be present in each position, not merely absent-by-deletion
  assert.equal((tools[0]!.args as { value: string }).value, MARK);
  const delta = (tools[0]!.result as { effect: { delta: { after: string } } }).effect.delta;
  assert.equal(delta.after, `value=${MARK}`);
  const messages = (gens[0]!.input as { messages: Record<string, unknown>[] }).messages;
  assert.match(String(JSON.stringify(messages[0])), /redacted:len=/);
  assert.match(String(JSON.stringify(messages[1])), /redacted:len=/);
  assert.match(JSON.stringify(gens[0]!.output), /redacted:len=/);
});

test('non-fill acts and unrelated content are untouched', async () => {
  const { telemetry, tools } = capture();
  const clickArgs = { verb: 'act', ref: 'n_abc123abc123', action: 'click' };
  const clickResult = { actRef: 'a_2', outcome: 'delivered',
    effect: { verdict: 'verified', evidence: 'navigation_post',
      delta: { before: 'http://a/', after: 'http://b/' } } };
  await telemetry.run(meta, async (t) => {
    t.toolCall({ verb: 'act', args: clickArgs, result: clickResult, ms: 1 });
  });
  assert.deepEqual(tools[0]!.args, clickArgs, 'redaction is structural, not a scrub of everything');
  assert.deepEqual(tools[0]!.result, clickResult);
});

// Position 6, found by the completeness critic: end({answer}) ships the model's
// final answer to the vendor, and the system prompt tells the model to report a
// field's value as "the page's verbatim string, units and all" — so an answer can
// echo what was typed. It cannot be keyed on STRUCTURE (an answer is prose), so it
// is closed by MEMORY: the redactor remembers what this episode filled and rewrites
// those exact strings. An exact match against a known secret, not a pattern guess.
test('a filled value echoed in the final answer is redacted', async () => {
  const ends: Record<string, unknown>[] = [];
  const trace: EpisodeTrace = {
    traceId: 'test',
    generation() { /* no-op */ },
    toolCall() { /* no-op */ },
    end(r) { ends.push(r as unknown as Record<string, unknown>); },
  };
  const telemetry = isolate({
    enabled: true,
    run: async (_meta, fn) => fn(trace),
    shutdown: async () => undefined,
  });

  await telemetry.run(meta, async (t) => {
    t.toolCall({ verb: 'act', args: fillArgs, result: fillResult, ms: 5 });
    t.end({ status: 'success', answer: `I entered ${SECRET} into the field`,
            modelCalls: 2, wirCalls: 2 });
  });

  const answer = String(ends[0]!['answer']);
  assert.ok(!answer.includes(SECRET), `the answer must not carry the value: ${answer}`);
  assert.equal(answer, `I entered ${MARK} into the field`);
});

// An answer that never contained a filled value must pass through untouched.
test('an unrelated answer is not rewritten', async () => {
  const ends: Record<string, unknown>[] = [];
  const telemetry = isolate({
    enabled: true,
    run: async (_meta, fn) => fn({
      traceId: 't', generation() {}, toolCall() {},
      end(r) { ends.push(r as unknown as Record<string, unknown>); },
    }),
    shutdown: async () => undefined,
  });
  await telemetry.run(meta, async (t) => {
    t.toolCall({ verb: 'act', args: fillArgs, result: fillResult, ms: 1 });
    t.end({ status: 'success', answer: '42', modelCalls: 1, wirCalls: 1 });
  });
  assert.equal(ends[0]!['answer'], '42');
});

test('WIR_TELEMETRY_LOG_VALUES=1 restores raw values', async () => {
  process.env['WIR_TELEMETRY_LOG_VALUES'] = '1';
  try {
    const { telemetry, tools } = capture();
    await telemetry.run(meta, async (t) => {
      t.toolCall({ verb: 'act', args: fillArgs, result: fillResult, ms: 5 });
    });
    assert.equal((tools[0]!.args as { value: string }).value, SECRET);
  } finally {
    delete process.env['WIR_TELEMETRY_LOG_VALUES'];
  }
});
