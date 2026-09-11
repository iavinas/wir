// Regression for the body-read timeout retry gap (docs/plans/fewer-misses.md
// §B3). postWithRetry used to return the Response with its body UNREAD; the
// caller parsed it outside the retry loop with the abort signal still armed, so
// a timeout firing mid-body was relabelled terminal "unparseable JSON" — 219
// discarded model calls over 6 episodes, `[provider] retry` appearing 0 times.
// The body read now lives inside the loop: a transport abort mid-body retries
// like any transport failure, while a fully-received malformed body stays
// terminal. The assertions are REQUEST COUNTS, not just outcomes — the defect
// was precisely a retry that never happened.
//
// Offline: no browser, no real provider — a scripted localhost mock, driving
// the REAL chatComplete.
import { strict as assert } from 'node:assert';
import { createServer, type Server, type ServerResponse } from 'node:http';
import { after, test } from 'node:test';

const GOOD = JSON.stringify({
  choices: [{ message: { role: 'assistant', content: 'ok' } }],
  usage: {
    prompt_tokens: 10, completion_tokens: 2,
    prompt_tokens_details: { cached_tokens: 3 },
    completion_tokens_details: { reasoning_tokens: 1 },
  },
});

type Handler = (res: ServerResponse) => void;
let script: Handler[] = [];
let requests = 0;
const server: Server = createServer((_req, res) => {
  requests += 1;
  const handler = script.shift();
  if (handler === undefined) { res.writeHead(500); res.end('script exhausted'); return; }
  handler(res);
});

// The body starts streaming, then never ends — the abort must fire MID-BODY,
// after fetch has already resolved with the response headers.
const hang: Handler = res => {
  res.writeHead(200, { 'content-type': 'application/json' });
  res.write('{"choices":[{"message":{"role":"assis');
};
const good: Handler = res => {
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(GOOD);
};
const notJson: Handler = res => {
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end('not json');
};
const badRequest: Handler = res => {
  res.writeHead(400, { 'content-type': 'application/json' });
  res.end('{"error":"bad request"}');
};

// provider.ts reads its env at module load, so the mock's port must exist and
// be in the environment BEFORE the first import. Loaded once, lazily.
let loaded: Promise<typeof import('../agent/provider.js')> | null = null;
function provider(): Promise<typeof import('../agent/provider.js')> {
  loaded ??= (async () => {
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
    const { port } = server.address() as { port: number };
    process.env['WIR_PROVIDER_BASE_URL'] = `http://127.0.0.1:${port}`;
    process.env['WIR_PROVIDER_API_KEY'] = 'test-key';
    process.env['WIR_MODEL'] = 'mock-model';
    process.env['WIR_PROVIDER_TIMEOUT_MS'] = '1000';
    return import('../agent/provider.js');
  })();
  return loaded;
}

after(() => { server.closeAllConnections(); server.close(); });

const messages = () => [{ role: 'user' as const, content: 'hi' }];

test('a mid-body timeout is transport: retried, second attempt wins — exactly 2 requests', async () => {
  const { chatComplete } = await provider();
  script = [hang, good];
  const before = requests;
  // The retry must also be VISIBLE: the measured symptom was zero retry lines.
  let retryLines = 0;
  const realWrite = process.stderr.write.bind(process.stderr);
  process.stderr.write = ((chunk: string | Uint8Array, ...rest: never[]) => {
    if (String(chunk).includes('[provider] retry')) retryLines += 1;
    return realWrite(chunk, ...rest);
  }) as typeof process.stderr.write;
  try {
    const result = await chatComplete(messages(), []);
    assert.equal(result.message.content, 'ok');
  } finally {
    process.stderr.write = realWrite;
  }
  assert.equal(requests - before, 2, 'the hung body costs one retry, no more');
  assert.equal(retryLines, 1, 'the retry is logged to stderr');
});

test('a fully-received `not json` body is terminal — exactly 1 request', async () => {
  const { chatComplete, ProviderError } = await provider();
  script = [notJson];
  const before = requests;
  await assert.rejects(() => chatComplete(messages(), []), (error: Error) => {
    assert.ok(error instanceof ProviderError);
    assert.match(error.message, /unparseable JSON/);
    return true;
  });
  assert.equal(requests - before, 1, 'a genuinely malformed body must not retry');
});

test('a 400 is terminal — exactly 1 request', async () => {
  const { chatComplete, ProviderError } = await provider();
  script = [badRequest];
  const before = requests;
  await assert.rejects(() => chatComplete(messages(), []), (error: Error) => {
    assert.ok(error instanceof ProviderError);
    assert.match(error.message, /provider returned 400/);
    return true;
  });
  assert.equal(requests - before, 1, 'other 4xx stays terminal');
});

test('happy path: one request, ChatResult fields exactly as the mock served them', async () => {
  const { chatComplete } = await provider();
  script = [good];
  const before = requests;
  const sent = messages();
  const result = await chatComplete(sent, []);
  assert.equal(requests - before, 1);
  assert.equal(result.message.content, 'ok');
  assert.equal(result.inputTokens, 10);
  assert.equal(result.outputTokens, 2);
  assert.equal(result.cachedTokens, 3);
  assert.equal(result.reasoningTokens, 1);
  assert.equal(result.reasoningText, null);
  // The telemetry seam: the exact request object, never a re-serialization.
  assert.equal((result.requestBody as { messages: unknown }).messages, sent);
});
