// Regression for the unseen-value flag (core/session.ts GateEligibleAct.unseen,
// ActReceipt.unseen), reproduced on the LIVE shopping container before this
// file existed (debug/probe_unseen_values.mjs, artifacts under
// debug/runs/probe/2026-09-03T*-unseen-values-*): task 521 ("Subscribe to the
// newsletter"), three arms, posted email=user@example.com twice and the
// operator's own address once — a value present in no page and no task — while
// the customer's own address sat on My Account, delivered by a plain `read {}`
// in the control run. The receipt showed the POST, the confrontation restated
// it, the model finished SUCCESS every time; the runtime knew exactly which
// text it had delivered and never said the value was in none of it.
//
// The rule is a delivery fact, the same class as "refs never observed this
// episode": a sent field value the caller TYPED into an <input> (a datum
// control; a <textarea> or contenteditable is prose the caller authors) with no
// internal whitespace, that no delivered text run and not the task instruction
// carried BEFORE the keystrokes. It flags; it never rejects, never judges the
// value, never reads an expected answer.
//
// An http fixture is justified as in act-receipt.test.ts: the live page proves
// the class but cannot hold its bodies still; the shapes are the live ones — a
// form POST that redirects, an email input beside a free-text body.
import { strict as assert } from 'node:assert';
import { createServer, type Server } from 'node:http';
import { test } from 'node:test';
import { WirSession } from '@wir/core';
import { mutateConfrontation } from '../agent/loop.js';

const FORM = `<form method="post" action="/subscribe">
    <label>Sign Up for Our Newsletter: <input type="email" name="email"></label>
    <label>Comment <textarea name="comment"></textarea></label>
    <input type="hidden" name="form_key" value="k1">
    <button>Subscribe</button>
  </form>`;
const PAGE = `<!doctype html><title>unseen</title><h1>Newsletter</h1>${FORM}`;
// The control: the page states the account's own address, as My Account does.
const SHOWN = `<!doctype html><title>unseen</title><h1>My Account</h1>
  <p>Contact Information</p><p>Emma Lopez emma.lopez@gmail.com</p>${FORM}`;

function serve(): Promise<{ url: string; close: () => void; posted: Record<string, string>[] }> {
  const posted: Record<string, string>[] = [];
  return new Promise((resolve) => {
    const server: Server = createServer((req, res) => {
      if (req.method === 'POST' && req.url === '/subscribe') {
        let body = '';
        req.on('data', (c: Buffer) => { body += c.toString(); });
        req.on('end', () => {
          posted.push(Object.fromEntries(new URLSearchParams(body)));
          res.writeHead(302, { location: '/' }); res.end();
        });
        return;
      }
      res.writeHead(200, { 'content-type': 'text/html' });
      res.end(req.url === '/shown' ? SHOWN : PAGE);
    });
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address() as { port: number };
      resolve({ url: `http://127.0.0.1:${addr.port}/`, posted, close: () => {
        (server as unknown as { closeAllConnections?: () => void }).closeAllConnections?.();
        server.close();
      } });
    });
  });
}

async function startSession(url: string, opts: { instruction?: string; deliveredBound?: number } = {}): Promise<WirSession> {
  const session = await WirSession.start({
    headless: true, expectedAction: 'MUTATE', storageStatePath: null,
    harPath: null, tracePath: null, debugScreenshots: false, ...opts,
  });
  await session.goto(url);
  return session;
}

async function refOf(session: WirSession, name: string, role: string): Promise<string> {
  const found = await session.dispatch({ verb: 'find', name, role });
  const ref = (found['matches'] as { ref: string }[])[0]?.ref;
  assert.ok(ref, `not found: ${role} ${name}: ${JSON.stringify(found)}`);
  return ref;
}

interface Acted { actRef: string; receipt: { unseen?: string[]; unseenUnchecked?: string;
  requests: { method: string; body?: { fields: Record<string, string> } }[] } }

async function subscribe(session: WirSession, email: string | null, comment: string | null): Promise<Acted> {
  if (email !== null) {
    const box = await refOf(session, 'newsletter', 'textbox');
    const filled = await session.dispatch({ verb: 'act', ref: box, action: 'fill', value: email });
    assert.equal((filled['effect'] as { verdict: string }).verdict, 'verified', JSON.stringify(filled));
  }
  if (comment !== null) {
    const body = await refOf(session, 'comment', 'textbox');
    const filled = await session.dispatch({ verb: 'act', ref: body, action: 'fill', value: comment });
    assert.equal((filled['effect'] as { verdict: string }).verdict, 'verified', JSON.stringify(filled));
  }
  const btn = await refOf(session, 'Subscribe', 'button');
  const acted = await session.dispatch({ verb: 'act', ref: btn, action: 'click' }) as unknown as Acted;
  const post = acted.receipt.requests.find(r => r.method === 'POST');
  assert.ok(post?.body, `the submit must be in the receipt: ${JSON.stringify(acted)}`);
  return acted;
}

test('an e-mail typed from nowhere is flagged on the receipt, in the ledger and in the confrontation; the finish is still accepted', async () => {
  const { url, close, posted } = await serve();
  const session = await startSession(url, { instruction: 'Subscribe to the newsletter' });
  try {
    const acted = await subscribe(session, 'user@example.com', null);
    assert.deepEqual(acted.receipt.unseen, ['email=user@example.com'], JSON.stringify(acted.receipt));
    assert.equal(acted.receipt.unseenUnchecked, undefined);
    assert.equal(posted[0]?.['email'], 'user@example.com');
    const ledger = session.gateEligibleActs();
    assert.equal(ledger.length, 1);
    assert.deepEqual(ledger[0]!.unseen, ['email=user@example.com']);
    assert.match(ledger[0]!.sent ?? '', /email=user@example\.com/);
    // form_key is a token field: the page minted it, and the rule never names it.
    assert.ok(!ledger[0]!.unseen!.some(u => u.startsWith('form_key')));
    const text = mutateConfrontation(session, '{}');
    assert.match(text, /values the page never showed you: email=user@example\.com/);
    assert.match(text, /A value the page never showed came from you; if the task meant the\naccount's own, read where the account states it\./);
    // A flag, never a rejection.
    const fin = await session.dispatch({ verb: 'finish', answer: '', evidenceRefs: [acted.actRef] });
    assert.equal(fin['accepted'], true, JSON.stringify(fin));
  } finally { await session.close(); close(); }
});

test('not flagged when the page displayed the address before it was typed', async () => {
  const { url, close } = await serve();
  const session = await startSession(`${url}shown`, { instruction: 'Subscribe to the newsletter' });
  try {
    // The page states the address, and the overview DELIVERS it — asserted, so
    // the absence of a flag below is proof of the rule, not of a missed run.
    const overview = await session.dispatch({ verb: 'read' });
    assert.match(JSON.stringify(overview), /emma\.lopez@gmail\.com/);
    const acted = await subscribe(session, 'emma.lopez@gmail.com', null);
    assert.equal(acted.receipt.unseen, undefined, JSON.stringify(acted.receipt));
    assert.equal(acted.receipt.unseenUnchecked, undefined);
    assert.equal(session.gateEligibleActs()[0]!.unseen, undefined);
    assert.doesNotMatch(mutateConfrontation(session, '{}'), /never showed/);
  } finally { await session.close(); close(); }
});

test('not flagged when the task instruction carries the value', async () => {
  const { url, close } = await serve();
  const session = await startSession(url, { instruction: 'Subscribe bob@example.org to the newsletter' });
  try {
    const acted = await subscribe(session, 'bob@example.org', null);
    assert.equal(acted.receipt.unseen, undefined, JSON.stringify(acted.receipt));
    assert.equal(session.gateEligibleActs()[0]!.unseen, undefined);
  } finally { await session.close(); close(); }
});

test('a free-text body is the caller\'s own words by design: never flagged', async () => {
  const { url, close, posted } = await serve();
  const session = await startSession(url, { instruction: 'Leave a comment' });
  try {
    const words = 'These words appear nowhere on the page and were typed on purpose.';
    const acted = await subscribe(session, null, words);
    // The body WAS sent — the rule declined to flag it, it did not miss it.
    assert.equal(posted[0]?.['comment'], words);
    assert.equal(acted.receipt.unseen, undefined, JSON.stringify(acted.receipt));
    assert.equal(acted.receipt.unseenUnchecked, undefined);
    assert.doesNotMatch(mutateConfrontation(session, '{}'), /never showed/);
  } finally { await session.close(); close(); }
});

test('when the delivered-text ledger overflowed, the flag says "not checked" instead of "never showed"', async () => {
  const { url, close } = await serve();
  // A bound the first find overflows; the product bound is 100,000 runs.
  const session = await startSession(url, { instruction: 'Subscribe to the newsletter', deliveredBound: 8 });
  try {
    await session.dispatch({ verb: 'read' });
    assert.ok(session.deliveredTextLedger().dropped > 0, JSON.stringify(session.deliveredTextLedger()));
    const acted = await subscribe(session, 'user@example.com', null);
    assert.equal(acted.receipt.unseen, undefined, JSON.stringify(acted.receipt));
    assert.match(acted.receipt.unseenUnchecked ?? '', /^not checked for email: the delivered-text ledger dropped \d+ runs past its 8-run bound/);
    const entry = session.gateEligibleActs()[0]!;
    assert.equal(entry.unseen, undefined);
    assert.match(entry.unseenUnchecked ?? '', /not checked for email/);
    const text = mutateConfrontation(session, '{}');
    assert.match(text, /\(not checked for email/);
    assert.doesNotMatch(text, /never showed/);
    const fin = await session.dispatch({ verb: 'finish', answer: '', evidenceRefs: [acted.actRef] });
    assert.equal(fin['accepted'], true, JSON.stringify(fin));
  } finally { await session.close(); close(); }
});
