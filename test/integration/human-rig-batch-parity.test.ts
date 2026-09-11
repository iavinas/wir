// The hand-driving rig must be able to express what the agent expresses.
//
// PROVEN DEFECT, found 2026-08-18 by driving reddit 600 by hand: the agent's
// system prompt devotes its two longest rules to batching ("ISSUE SEVERAL TOOL
// CALLS AT ONCE", "FILLING A FORM IS ONE TURN, NOT ONE TURN PER FIELD"),
// agent/loop.ts executes a batch until a call moves the page, and
// test/batched-tool-calls.test.ts pins that behaviour. But wir-cli/wir_cli.mjs
// rejected a JSON array with `tool call needs a "name"`, and
// wir-cli/human_provider.mjs hardcoded `tool_calls: [ ...one... ]`.
//
// So every one of the 764 episodes in the hand-driven sweep corpus ran ONE CALL
// PER TURN while being instructed to batch. That silently inflated every verb
// count in the corpus and made its economy numbers incomparable to a provider
// run — a measurement defect, not merely an ergonomic one.
//
// This is the same drift class as test/action-list-drift.test.ts (the enum the
// prose disagreed with for months). The rule it pins: when the agent's seam
// gains a capability, the rig that stands in for a model must gain it too, or
// the rig stops being a measurement of the agent.
//
// NO FIXTURE PAGE and no browser: this is a protocol-shape defect, decidable
// from the shim's own wire output.
import { strict as assert } from 'node:assert';
import { spawn } from 'node:child_process';
import { test } from 'node:test';

const PORT = 8917; // not 8899: never collide with a live hand-driven session

// /tools answers immediately; /pending LONG-POLLS until a turn exists, so it
// must never be used as a readiness probe (doing so deadlocks the test).
function waitFor(url: string, tries = 60): Promise<void> {
  return new Promise((resolve, reject) => {
    const attempt = (n: number) => {
      fetch(url).then(() => resolve()).catch(() => {
        if (n <= 0) reject(new Error(`shim never came up at ${url}`));
        else setTimeout(() => attempt(n - 1), 100);
      });
    };
    attempt(tries);
  });
}

test('a batch submitted through the rig reaches the agent as several tool_calls', async () => {
  const shim = spawn(process.execPath,
    ['wir-cli/human_provider.mjs', '--port', String(PORT), '--session', 'batch-parity'],
    { stdio: 'ignore' });
  try {
    await waitFor(`http://127.0.0.1:${PORT}/tools`);

    // The agent asks for a decision. Do not await yet — the shim blocks this
    // request until a driver submits, which is the whole point of the rig.
    const completion = fetch(`http://127.0.0.1:${PORT}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'human',
        messages: [{ role: 'user', content: 'turn 1' }],
        tools: [{ type: 'function', function: { name: 'find', parameters: {} } }],
      }),
    });

    // The driver answers with TWO independent calls, as the prompt instructs.
    // /pending blocks until the shim has captured the agent's turn, which is
    // exactly the synchronisation the driver relies on.
    const pending = await (await fetch(`http://127.0.0.1:${PORT}/pending`)).json() as { turn?: number };
    assert.ok(pending.turn, 'the shim must present the agent turn to the driver');
    // NOT awaited: /submit answers with the NEXT turn, which never arrives here
    // because nothing is driving the agent loop. Awaiting it deadlocks the test
    // — the shim behaving exactly as the driver needs it to.
    const submitted = fetch(`http://127.0.0.1:${PORT}/submit`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify([
        { name: 'find', arguments: { role: 'link', name: 'Forums' } },
        { name: 'find', arguments: { name: 'Submit' } },
      ]),
    });
    submitted.catch(() => undefined);

    // Bounded: on the PRE-FIX shim the batch is refused with 400, `pending` is
    // never cleared and the completion never resolves. Without this the test
    // hangs instead of failing, and a tripwire that hangs is a bad tripwire.
    const settled = await Promise.race([
      completion,
      new Promise<null>((r) => setTimeout(() => r(null), 8000)),
    ]);
    assert.ok(settled, 'the agent never received a completion — the rig refused the batch');
    const body = await settled.json() as {
      choices: { message: { tool_calls?: { function: { name: string; arguments: string } }[] } }[];
    };
    const calls = body.choices[0]?.message?.tool_calls ?? [];
    assert.equal(calls.length, 2,
      'both calls must reach the agent — one is the defect this test exists for');
    assert.deepEqual(calls.map((c) => c.function.name), ['find', 'find']);
    assert.equal(JSON.parse(calls[1]!.function.arguments)['name'], 'Submit',
      'the second call must keep its own arguments');
  } finally {
    shim.kill('SIGKILL');
  }
});

// THE LOOP'S APPENDED LINES REACH THE DRIVER'S WINDOW. agent/loop.ts appends
// the next-call offers (agent/offers.ts) — and the repeat-rejection escalation
// and the batching note before them — AFTER the verb's JSON, inside the tool
// result. The model receives one string; the driver must be shown the same
// string with the appended call still legible and copyable, or the rig is
// measuring a consumer that was never offered it. Same protocol-shape class as
// the batch above: decidable from the shim's wire and the CLI's stdout alone.
test('a next-call offer appended to a tool result is shown to the driver, copyable', async () => {
  const port = PORT + 1;
  const shim = spawn(process.execPath,
    ['wir-cli/human_provider.mjs', '--port', String(port), '--session', 'offer-parity'],
    { stdio: 'ignore' });
  try {
    await waitFor(`http://127.0.0.1:${port}/tools`);
    const offer = 'Waiting for something? Say it on the act: '
      + '{"verb":"act","ref":"n_84e04e750cca","action":"scroll","value":"end","until":{"text":"<the words you expect>"}}';
    const result = JSON.stringify({ documentEpoch: 'E', actRef: 'a_6',
      effect: { verdict: 'verified', evidence: 'scrolled_no_new_content' } });
    const completion = fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'human',
        messages: [
          { role: 'system', content: 'sys' }, { role: 'user', content: 'task' },
          { role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function',
            function: { name: 'act', arguments: '{"ref":"n_84e04e750cca","action":"scroll","value":"end"}' } }] },
          { role: 'tool', tool_call_id: 'c1', content: `${result}\n${offer}` },
        ],
        tools: [{ type: 'function', function: { name: 'act', parameters: {} } }],
      }),
    });
    completion.catch(() => undefined);
    await (await fetch(`http://127.0.0.1:${port}/pending`)).json();
    const peek = spawn(process.execPath, ['wir-cli/wir_cli.mjs', 'peek'],
      { env: { ...process.env, WIR_HUMAN_PORT: String(port) }, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    peek.stdout.on('data', (d: Buffer) => { out += d.toString(); });
    await new Promise<void>((resolve) => peek.on('close', () => resolve()));
    assert.ok(out.includes('"actRef": "a_6"'),
      `the verb's JSON is still pretty-printed with the offer behind it:\n${out}`);
    assert.ok(out.includes(offer), `the offer line reaches the driver verbatim:\n${out}`);
  } finally {
    shim.kill('SIGKILL');
  }
});
