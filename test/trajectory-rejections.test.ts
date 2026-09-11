// Regression for review finding B8: the trajectory lost exactly the calls that
// went wrong. record() fires only inside wirDispatch, so a tool call the loop
// refuses BEFORE dispatch — garbled JSON arguments, an unknown tool name, a
// budget refusal — reached the model and left no trace at all. The debug plane
// is supposed to be lossless (lessons.md preservation invariant 6); a garbled
// argument is only diagnosable in the form the model actually emitted.
//
// The agent binary is run for real against a local OpenAI-compatible provider
// stub — no fake inside production code, no private copy of the loop.
import { strict as assert } from 'node:assert';
import { spawn } from 'node:child_process';
import { createServer, type Server } from 'node:http';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const REPO = fileURLToPath(new URL('../..', import.meta.url));

/** Serves a scripted list of assistant messages, one per chat completion. A turn
 *  may be a function of the literal request body, for the one thing a static
 *  script cannot do: cite a ref the runtime minted this episode. Refs are opaque
 *  and runtime-minted, so a test that hardcoded one would be asserting on an
 *  identity the runtime is free to change — the stub reads it out of its own
 *  transcript, exactly as the model does. */
export async function providerStub(turns: unknown[]): Promise<{ url: string; close: () => Promise<void> }> {
  let turn = 0;
  const server: Server = createServer((req, res) => {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', () => {
      const scripted = turns[Math.min(turn++, turns.length - 1)];
      const message = typeof scripted === 'function'
        ? (scripted as (body: string) => unknown)(body) : scripted;
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({
        choices: [{ message }],
        usage: { prompt_tokens: 1000, completion_tokens: 10 },
      }));
    });
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as { port: number }).port;
  return {
    url: `http://127.0.0.1:${port}/v1`,
    close: () => new Promise<void>(resolve => { server.close(() => resolve()); }),
  };
}

export function toolCall(name: string, args: string): unknown {
  return { role: 'assistant', content: null,
    tool_calls: [{ id: `call_${name}`, type: 'function', function: { name, arguments: args } }] };
}

/** A plain text turn — what the completion contract elicits, and what a scripted
 *  evaluator verdict looks like. */
export function textReply(content: string): unknown {
  return { role: 'assistant', content };
}

/** The scripted contract reply a RETRIEVE/NAVIGATE episode consumes as its first
 *  model call whenever maxModelCalls > 3. Tests that budget 4+ calls prepend it. */
export const CONTRACT_REPLY = textReply(
  '- the set the question names, bounded by the page\n'
  + '- no ordering assumed\n- empty kept distinct from missing\n- answer as asked');

export interface RunResult {
  response: Record<string, unknown>;
  trajectory: Record<string, unknown>[];
  /** provider-payloads.jsonl — what the model ACTUALLY received. The trajectory
   *  cannot answer that: wirDispatch records its entry BEFORE the payload is
   *  assembled, so likelyTargets, the site-skill block and the task reminder are
   *  all appended afterwards. Anything asserting on what reached the model has to
   *  read this. */
  payloads: Record<string, unknown>[];
}

/** Runs the built agent end-to-end over the stdin/stdout protocol. */
export async function runAgent(opts: {
  turns: unknown[]; instruction?: string; expectedAction?: 'RETRIEVE' | 'MUTATE';
  maxModelCalls?: number; maxWirCalls?: number; maxInputTokensPerCall?: number;
  env?: Record<string, string>; html?: string; uploadDir?: string;
}): Promise<RunResult> {
  const stub = await providerStub(opts.turns);
  const dir = mkdtempSync(join(tmpdir(), 'wir-agent-'));
  writeFileSync(join(dir, 'a.html'),
    opts.html ?? '<!doctype html><title>a</title><h1>Host</h1><a href="/x">a link</a>');
  const request = {
    protocolVersion: 'wa-benchmark-agent-1',
    runId: 'run-trajectory-1',
    task: {
      benchmark: 'test', taskId: 1, revision: 1,
      instruction: opts.instruction ?? 'do nothing',
      startUrl: `file://${dir}/a.html`,
      expectedAction: opts.expectedAction ?? 'RETRIEVE',
    },
    authority: { riskMode: 'read_only' },
    browser: {
      headless: true, harPath: join(dir, 'network.har'), tracePath: join(dir, 'trace.zip'),
      storageStatePath: null, slowMoMs: 0,
      ...(opts.uploadDir !== undefined ? { uploadDir: opts.uploadDir } : {}),
    },
    budgets: {
      maxWallTimeMs: 120_000, maxModelCalls: opts.maxModelCalls ?? 2,
      maxWirCalls: opts.maxWirCalls ?? 50,
      ...(opts.maxInputTokensPerCall !== undefined
        ? { maxInputTokensPerCall: opts.maxInputTokensPerCall } : {}),
    },
    artifacts: { trajectoryPath: join(dir, 'trajectory.jsonl'), metricsPath: join(dir, 'metrics.json') },
  };
  const child = spawn('node', [join(REPO, 'dist/agent/main.js')], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      PATH: process.env['PATH'] ?? '', HOME: process.env['HOME'] ?? '',
      WIR_PROVIDER_BASE_URL: stub.url, WIR_PROVIDER_API_KEY: 'test', WIR_MODEL: 'stub',
      ...opts.env,
    },
  });
  let stdout = '';
  child.stdout.on('data', d => { stdout += d; });
  child.stderr.on('data', () => { /* the log stream; not asserted here */ });
  child.stdin.write(JSON.stringify(request));
  child.stdin.end();
  await new Promise<void>(resolve => child.on('close', () => resolve()));
  await stub.close();
  return {
    response: JSON.parse(stdout) as Record<string, unknown>,
    trajectory: readFileSync(request.artifacts.trajectoryPath, 'utf8')
      .split('\n').filter(Boolean).map(l => JSON.parse(l) as Record<string, unknown>),
    payloads: (() => {
      try {
        return readFileSync(join(dir, 'provider-payloads.jsonl'), 'utf8')
          .split('\n').filter(Boolean).map(l => JSON.parse(l) as Record<string, unknown>);
      } catch { return []; }
    })(),
  };
}

test('a refused tool call is recorded with its arguments verbatim', async () => {
  const GARBLED = '{"name": "unterminated';
  const { trajectory } = await runAgent({
    turns: [toolCall('find', GARBLED), toolCall('teleport', '{}')],
    maxModelCalls: 2,
  });

  const rejected = trajectory.filter(r => r['kind'] === 'rejected');
  assert.equal(rejected.length, 2,
    `both refused calls must appear in the trajectory: ${JSON.stringify(trajectory)}`);

  const garbled = rejected.find(r => r['name'] === 'find');
  assert.ok(garbled, JSON.stringify(rejected));
  assert.equal(garbled['rawArguments'], GARBLED,
    'the argument string must be recorded exactly as the model emitted it');
  assert.match(String(garbled['reason']), /not valid JSON/);

  const unknown = rejected.find(r => r['name'] === 'teleport');
  assert.ok(unknown, JSON.stringify(rejected));
  assert.match(String(unknown['reason']), /unknown tool/);
});
