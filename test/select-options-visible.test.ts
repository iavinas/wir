// A select's own options must be visible to the caller that has to choose one.
//
// This is the zero-tolerance class in CLAUDE.md — "the runtime saw it and did not
// show you" — and it was a contradiction inside this system. agent/loop.ts told the
// model to "enumerate the chooser's own options (find with role option over the
// select)", while core/compiler.ts documented that <option> elements are never
// compiled. The instruction named a query the compiler guaranteed to answer with
// nothing.
//
// Reproduced in the probe first, on the live shopping storefront
// (http://localhost:7770/sales/guest/form/, a two-option select):
//     find(role=option)   -> matched 0 of 117 searched
//     read(the select)    -> childrenTotal 0, children 0
//     page_ground_truth   -> contains "Email": true, "ZIP Code": true
// The oracle is a direct page.evaluate with no WIR in it, so that last line is the
// runtime being contradicted by the page itself.
//
// Found by driving five shopping_admin tasks with a strong model after a small one
// had spiralled on them. On task 201 find(role=option) matched 0 of 5977 and
// find(name="Suspected Fraud") matched 0, while the option demonstrably existed;
// only act(select) could reveal it. The strong model guessed past it. Its own note
// is why this matters: "a weaker model would read that empty find as 'there is no
// such status' and stop — which is very close to how the haiku run died."
//
// The options still get no refs. Inside a closed <select> they are genuinely not
// separately addressable and `act` selects by LABEL, so the labels travel on the
// select — the node the caller acts on — in BOTH find and read.
import { strict as assert } from 'node:assert';
import { createServer, type Server } from 'node:http';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { WirSession } from '@wir/core';

// Two selects, as on the real page: one REAL chooser and one styled decoy with the
// same role. Telling them apart is what optionCount was built for; choosing from
// the real one is what optionLabels is for.
const PAGE = `<!doctype html><title>choose</title><h1>Find your order</h1>
  <label for="t">Find Order By</label>
  <select id="t" name="oar_type">
    <option value="email">Email</option>
    <option value="zip">ZIP Code</option>
    <option value="order">Order Number</option>
  </select>
  <label for="d">Search</label>
  <div id="d" role="combobox" tabindex="0">a styled wrapper, not a select</div>`;

function serve(): Promise<{ server: Server; url: string; close: () => void }> {
  return new Promise((resolve) => {
    const server = createServer((_req, res) => {
      res.writeHead(200, { 'content-type': 'text/html' });
      res.end(PAGE);
    });
    server.listen(0, '127.0.0.1', () => {
      const port = (server.address() as { port: number }).port;
      resolve({ server, url: `http://127.0.0.1:${port}/`, close: () => server.close() });
    });
  });
}

async function session(): Promise<WirSession> {
  const dir = mkdtempSync(join(tmpdir(), 'wir-opts-'));
  return WirSession.start({
    headless: true, expectedAction: 'RETRIEVE', storageStatePath: null,
    harPath: join(dir, 'network.har'), tracePath: join(dir, 'trace.zip'),
    debugScreenshots: false,
  });
}

test('a select carries its option labels in read, and the decoy does not', async () => {
  const site = await serve();
  const s = await session();
  try {
    await s.goto(site.url);
    const page = await s.dispatch({ verb: 'read' });
    const controls = page['controls'] as Record<string, unknown>[];
    const real = controls.find(c => String(c['name'] ?? '').includes('Find Order By'));
    assert.ok(real, `the select must compile: ${JSON.stringify(controls).slice(0, 400)}`);
    assert.deepEqual(real['optionLabels'], ['Email', 'ZIP Code', 'Order Number'],
      'the caller has to choose one of these, so it has to be able to read them: '
      + JSON.stringify(real));
    assert.equal(real['optionCount'], 3);

    // THE CONTROL. A styled div with role=combobox owns no options, and saying it
    // did would make optionCount useless for its original purpose — telling the
    // real chooser from the wrapper beside it.
    const decoy = controls.find(c => String(c['name'] ?? '').includes('Search')
      || String(c['name'] ?? '').includes('styled'));
    if (decoy) {
      assert.equal(decoy['optionLabels'], undefined,
        `a wrapper with no <option> children must not claim any: ${JSON.stringify(decoy)}`);
    }
  } finally {
    await s.close();
    site.close();
  }
});

// find is where the system prompt sends a caller looking for a chooser, so a
// select whose options are invisible THERE is one the caller cannot choose from —
// which is exactly how this was found.
test('find carries them too, so the caller need not know which verb to ask', async () => {
  const site = await serve();
  const s = await session();
  try {
    await s.goto(site.url);
    const found = await s.dispatch({ verb: 'find', role: 'combobox' });
    const matches = found['matches'] as Record<string, unknown>[];
    const real = matches.find(m => String(m['name'] ?? '').includes('Find Order By'));
    assert.ok(real, `find must reach the select: ${JSON.stringify(found).slice(0, 400)}`);
    assert.deepEqual(real['optionLabels'], ['Email', 'ZIP Code', 'Order Number'],
      `find must show the options too: ${JSON.stringify(real)}`);
  } finally {
    await s.close();
    site.close();
  }
});

// The labels are not a substitute for acting: what the caller reads must be what
// `act` accepts, or we have replaced an invisible chooser with a misleading one.
test('a label read off the node is a label act(select) accepts', async () => {
  const site = await serve();
  const s = await session();
  try {
    await s.goto(site.url);
    const found = await s.dispatch({ verb: 'find', role: 'combobox' });
    const matches = found['matches'] as Record<string, unknown>[];
    const real = matches.find(m => String(m['name'] ?? '').includes('Find Order By'))!;
    const label = (real['optionLabels'] as string[])[1]!;   // 'ZIP Code'

    const act = await s.dispatch({
      verb: 'act', action: 'select', ref: String(real['ref']), value: label,
    });
    const effect = act['effect'] as { verdict: string; evidence: string };
    assert.equal(effect.verdict, 'verified',
      `the label the runtime showed must be one act accepts: ${JSON.stringify(act).slice(0, 400)}`);
    assert.equal(effect.evidence, 'option_selected');
  } finally {
    await s.close();
    site.close();
  }
});
