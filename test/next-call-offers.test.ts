// The next-call offer builder (agent/offers.ts) over a recorded trajectory
// slice — the three result records that carry the patterns it exists for,
// taken verbatim from develop-v2 arm 4 (wir-run-final-v2-arm4:
// run-write-shopping_admin/task-464/attempt-1 rows 20/24, run-write-gitlab/
// task-660/attempt-1 rows 20/23/87). Pure function, no browser, no provider:
// the defect is "the runtime's result showed the pattern and the model was
// handed nothing it could copy", and that is decidable from the records alone.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import {
  createOfferState, offerAfter, sentExpectCall, OFFER_MAX_BYTES, OFFER_IGNORE_LIMIT,
} from '../agent/offers.js';

const ENV = { documentEpoch: 'E', freshness: 'dirty', coverageIncomplete: false, outcome: 'delivered' };
const scrolledEnd = { ...ENV, actRef: 'a_4', effect: { verdict: 'verified', evidence: 'scrolled',
  delta: { before: 'scrollTop=0', after: 'scrollTop=4441' } },
  receipt: { attribution: 'unarmed', windowMs: 0, total: 0, requests: [] } };
const scrolledNoNew = (actRef: string) => ({ ...ENV, actRef,
  effect: { verdict: 'verified', evidence: 'scrolled_no_new_content',
    delta: { before: 'scrollTop=4441 (already at the end)', after: 'scrollTop=4441' } },
  receipt: { attribution: 'unarmed', windowMs: 0, total: 0, requests: [] } });
// 660 row 20: the title fill, then row 23: read {} to see whether it took.
const fillArgs = { ref: 'n_63f1310a5850', action: 'fill', value: 'add support for oh-my-zsh' };
const filled = { ...ENV, actRef: 'a_5', effect: { verdict: 'verified', evidence: 'value_set',
  delta: { before: 'value=""', after: 'value="add support for oh-my-zsh"' } },
  receipt: { attribution: 'window', windowMs: 438, total: 0, requests: [] } };
// 660 row 87: the submit click — receipt carries the POST with the filled title.
const submitArgs = { ref: 'n_af01e6dc4c2e', action: 'click' };
const submitted = { ...ENV, actRef: 'a_23', effect: { verdict: 'verified', evidence: 'navigation_post',
  delta: { before: 'http://localhost:8023/x/-/issues/new', after: 'http://localhost:8023/x/-/issues/1 [POST request answered 302]' } },
  receipt: { attribution: 'window', windowMs: 3849, total: 57, requests: [
    { atMs: 30, type: 'document', method: 'POST', url: 'http://localhost:8023/byteblaze/dotfiles/-/issues',
      status: 302, initiator: 'other', body: { encoding: 'form', fields: {
        authenticity_token: 'wu4imOjA6Q8TZPWt8WYZ', 'issue[title]': 'add support for oh-my-zsh',
        'issue[description]': '', 'issue[confidential]': '0', 'issue[due_date]': '07/18/2033' } } },
    { atMs: 2775, type: 'fetch', method: 'POST', url: 'http://localhost:8023/api/graphql', status: 200, initiator: 'script' },
  ] } };
const bytes = (s: string) => Buffer.byteLength(s, 'utf8');

test('the second consecutive no-new-content scroll on one target is offered until, once', () => {
  const st = createOfferState();
  const ref = 'n_84e04e750cca';
  assert.equal(offerAfter(st, 'act', { ref, action: 'scroll', value: 'end' }, scrolledEnd).offer, null,
    'a scroll that moved offers nothing');
  assert.equal(offerAfter(st, 'act', { ref, action: 'scroll' }, scrolledNoNew('a_5')).offer, null,
    'the FIRST no-new-content answer is the runtime already saying so');
  // A read in between does not break the chain: the model is re-scrolling, not acting elsewhere.
  offerAfter(st, 'read', {}, { ...ENV, regions: [] });
  const second = offerAfter(st, 'act', { ref, action: 'scroll', value: 'end' }, scrolledNoNew('a_6'));
  assert.ok(second.offer, 'the second consecutive one is the pattern');
  assert.equal(second.offer.pattern, 'until');
  assert.ok(second.offer.line.includes(`{"verb":"act","ref":"${ref}","action":"scroll","value":"end","until":{"text":"<the words you expect>"}}`),
    `the literal call, same ref, same action, placeholder marked: ${second.offer.line}`);
  assert.ok(!second.offer.line.includes('network'), 'network idle is refused on scroll, so it is not offered there');
  assert.ok(second.offer.line.includes('"until":{"gone":'), 'the alternative is one scroll accepts');
  assert.ok(bytes(second.offer.line) <= OFFER_MAX_BYTES, `${bytes(second.offer.line)} bytes`);
  const third = offerAfter(st, 'act', { ref, action: 'scroll', value: 'end' }, scrolledNoNew('a_7'));
  assert.equal(third.offer, null, 'one offer per target, then silence');
  assert.deepEqual(third.outcome, { pattern: 'until', taken: false, ignored: 1 },
    'the act after the offer did not carry until: ignored once');
});

test('an unknown verdict is offered until with network idle as the alternative; taken is counted', () => {
  const st = createOfferState();
  const unmoved = { ...ENV, actRef: 'a_1', effect: { verdict: 'unknown', evidence: 'no_observable_change_yet' },
    receipt: { attribution: 'window', windowMs: 400, total: 0, requests: [] } };
  const r = offerAfter(st, 'act', { ref: 'n_4cf334b528c9', action: 'click' }, unmoved);
  assert.ok(r.offer && r.offer.line.includes('"until":{"network":"idle"}'), r.offer?.line);
  const next = offerAfter(st, 'act', { ref: 'n_4cf334b528c9', action: 'click', until: { network: 'idle' } },
    { ...unmoved, actRef: 'a_2' });
  assert.deepEqual(next.outcome, { pattern: 'until', taken: true, ignored: 0 });
  assert.equal(next.offer, null, 'an act that already carries until is never offered one');
});

test('a read straight after a value_set fill is offered expect.state.value shaped from that fill, once per shape', () => {
  const st = createOfferState();
  assert.equal(offerAfter(st, 'act', fillArgs, filled).offer, null, 'the fill itself is not the pattern');
  // The field's own region, as the first episodes on this build read it — not only the overview.
  const r = offerAfter(st, 'read', { target: 'n_8066233b44cf' }, { ...ENV, regions: [] });
  assert.ok(r.offer, 'a read after a local-only act is the pattern');
  assert.equal(r.offer.pattern, 'expect');
  assert.ok(r.offer.line.endsWith(
    '{"verb":"act","ref":"n_63f1310a5850","action":"fill","value":"add support for oh-my-zsh","expect":{"state":{"value":"add support for oh-my-zsh"}}}'),
    r.offer.line);
  assert.ok(bytes(r.offer.line) <= OFFER_MAX_BYTES);
  // A cursor is a continuation of a read already in hand, not a check.
  offerAfter(st, 'act', fillArgs, { ...filled, actRef: 'a_6' });
  assert.equal(offerAfter(st, 'read', { cursor: 'c_50' }, { ...ENV }).offer, null);
  // The same shape is not offered twice in an episode.
  offerAfter(st, 'act', fillArgs, { ...filled, actRef: 'a_7' });
  assert.equal(offerAfter(st, 'read', {}, { ...ENV }).offer, null, 'once per shape');
});

test('after two ignored offers of a pattern the model is not shown a third', () => {
  const st = createOfferState();
  for (let i = 0; i < OFFER_IGNORE_LIMIT + 1; i += 1) {
    const ref = `n_${i}`;
    const unmoved = { ...ENV, actRef: `a_${i}`, effect: { verdict: 'unknown', evidence: 'no_observable_change_yet' } };
    const r = offerAfter(st, 'act', { ref, action: 'click' }, unmoved);
    if (i < OFFER_IGNORE_LIMIT) assert.ok(r.offer, `offer ${i + 1} is shown`);
    else assert.equal(r.offer, null, 'the third is not: the model walked past two');
  }
});

test('the confrontation repair is the literal expect.sent built from the cited act\'s own receipt', () => {
  const line = sentExpectCall(submitArgs, submitted, new Set(['add support for oh-my-zsh', '07/18/2033']));
  // Both filled fields would make 203 bytes; the cap drops the LAST one and
  // keeps the path and the first field whole — never a cut value.
  assert.equal(line,
    '{"verb":"act","ref":"n_af01e6dc4c2e","action":"click","expect":{"sent":{"method":"POST","path":"/byteblaze/dotfiles/-/issues","fields":{"issue[title]":"add support for oh-my-zsh"}}}}');
  assert.ok(bytes(line!) <= OFFER_MAX_BYTES, `${bytes(line!)} bytes`);
  assert.ok(!line!.includes('authenticity_token'), 'only the fields the model itself filled');
  const shorter = sentExpectCall(submitArgs, submitted, new Set(['07/18/2033']));
  assert.ok(shorter!.includes('"fields":{"issue[due_date]":"07/18/2033"}'), `a field that fits is kept: ${shorter}`);
  assert.equal(sentExpectCall(fillArgs, filled, new Set()), null, 'a receipt with no non-GET request builds nothing');
  // A long value drops fields from the end rather than cutting the path or a value.
  const long = 'x'.repeat(150);
  const longSubmit = { ...submitted, receipt: { ...submitted.receipt, requests: [
    { ...submitted.receipt.requests[0], body: { encoding: 'form', fields: { 'issue[title]': long } } } ] } };
  const fitted = sentExpectCall(submitArgs, longSubmit, new Set([long]));
  assert.ok(fitted && bytes(fitted) <= OFFER_MAX_BYTES && fitted.includes('"path":"/byteblaze/dotfiles/-/issues"')
    && !fitted.includes('fields'), fitted ?? 'null');
});
