// `scroll` exists for exactly one thing — reaching content a page renders only
// as you scroll — and its verdict was decided by a signal that is constant on
// exactly that case.
//
// `newContent` tested a rising element count (`querySelectorAll('*').length`)
// or a growing `scrollHeight`. A virtualized list recycles a FIXED number of
// nodes behind a full-height spacer, so both are false while every row on
// screen is new. The act then answered `verified/scrolled_no_new_content`, and
// agent/loop.ts turns that token into "stop, and stop when it reports none" —
// so the model stops one viewport into a list it has never read.
//
// core/act.ts already documented both halves ("react-virtualized shows 50,000px
// of scroll extent behind 81 rendered children"; "a virtualized list sizes its
// scrollbar with a full-height spacer, so the obvious tell reads complete on
// exactly the case it was meant to catch") — and the verdict was still built on
// them.
//
// FIXTURE JUSTIFIED: the condition is a scroller whose node count and
// scrollHeight are both pinned while its text changes. No real site can be asked
// to hold that still, and the recorded corpus has no virtualized page. One page,
// one proven condition.
import { strict as assert } from 'node:assert';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { WirSession } from '@wir/core';

// 12 rows, recycled. scrollHeight is a fixed spacer. Element count never moves.
const VIRTUAL = `<!doctype html><title>virtual</title><h1>Feed</h1>
<div id="list" style="height:240px;overflow-y:auto;position:relative">
  <div id="spacer" style="height:20000px;position:absolute;top:0;left:0;width:1px"></div>
  <div id="rows" style="position:sticky;top:0"></div>
</div>
<script>
  const list = document.getElementById('list');
  const rows = document.getElementById('rows');
  const ROW = 20, N = 12;
  function render() {
    const first = Math.floor(list.scrollTop / ROW);
    // Reuse the SAME 12 nodes; only their text changes. This is what a windowed
    // list does, and it is why counting nodes cannot see new content.
    while (rows.children.length < N) rows.appendChild(document.createElement('div'));
    for (let k = 0; k < N; k++) rows.children[k].textContent = 'row ' + (first + k);
  }
  list.addEventListener('scroll', render);
  render();
</script>`;

async function start(html: string): Promise<WirSession> {
  const dir = mkdtempSync(join(tmpdir(), 'wir-vscroll-'));
  writeFileSync(join(dir, 'a.html'), html);
  const session = await WirSession.start({
    headless: true, expectedAction: 'RETRIEVE', storageStatePath: null,
  });
  await session.goto(`file://${dir}/a.html`);
  return session;
}

const effectOf = (r: Record<string, unknown>): { verdict: string; evidence: string;
  delta: { before: string; after: string } } =>
  r['effect'] as { verdict: string; evidence: string; delta: { before: string; after: string } };

test('a virtualized list that recycles its nodes still reports new content', async () => {
  const session = await start(VIRTUAL);
  try {
    const found = await session.dispatch({ verb: 'find', name: 'row 0' });
    const ref = ((found['matches'] as { ref: string }[]) ?? [])[0]?.ref;
    assert.ok(ref, `needed a ref inside the list: ${JSON.stringify(found)}`);

    const page = session.host.page;
    const before = await page.evaluate(() =>
      (document.getElementById('rows') as HTMLElement).innerText);
    const countBefore = await page.evaluate(() =>
      (document.getElementById('list') as HTMLElement).querySelectorAll('*').length);

    const acted = await session.dispatch({ verb: 'act', ref, action: 'scroll' });
    const effect = effectOf(acted);

    const after = await page.evaluate(() =>
      (document.getElementById('rows') as HTMLElement).innerText);
    const countAfter = await page.evaluate(() =>
      (document.getElementById('list') as HTMLElement).querySelectorAll('*').length);

    // The fixture must actually be virtualized, or this pins nothing: the rows
    // changed and the node count did not.
    assert.notEqual(after, before, 'the fixture must recycle rows into new text');
    assert.equal(countAfter, countBefore, 'the fixture must hold its node count fixed');

    assert.equal(effect.verdict, 'verified', JSON.stringify(acted));
    assert.equal(effect.evidence, 'scrolled',
      `a screen of never-seen rows must not read as "no new content": ${JSON.stringify(effect)}`);
  } finally { await session.close(); }
});

test('a scroller reports when it has reached its end, not only "nothing new"', async () => {
  // atEnd was computed on BOTH sides and read only on `before`, so after a
  // scroll a caller could not tell "exhausted" from "moved but revealed
  // nothing" — two states that want different next moves.
  const SHORT = `<!doctype html><title>short</title>
    <div id="list" style="height:60px;overflow-y:auto">
      <p>line one</p><p>line two</p><p>line three</p><p>line four</p><p>line five</p></div>`;
  const session = await start(SHORT);
  try {
    const found = await session.dispatch({ verb: 'find', name: 'line one' });
    const ref = ((found['matches'] as { ref: string }[]) ?? [])[0]?.ref;
    assert.ok(ref, JSON.stringify(found));
    let sawEnd = false;
    let last = '';
    for (let i = 0; i < 12 && !sawEnd; i += 1) {
      const acted = await session.dispatch({ verb: 'act', ref, action: 'scroll' });
      last = effectOf(acted).delta.after;
      sawEnd = /at the end of this container/.test(last);
    }
    assert.ok(sawEnd, `scrolling to the bottom must say so; last delta was ${last}`);
  } finally { await session.close(); }
});

// The other virtualization style, and the one that broke the measurement.
//
// react-window DESTROYS and recreates its rows rather than reusing them, so the
// element `scroll` was called on is gone by the time the verdict is computed.
// `describeScroll` re-walked the ancestor chain on both sides, and a detached
// row has no chain — so the second reading fell through to
// `document.scrollingElement` and described THE PAGE.
//
// Observed before the fix:
//   before : <div#list> ... extent=20000x1264
//   after  :            ... extent=900x1280 [at the end of this container]
//
// Two lies in one delta: the numbers belong to a different box, and the page
// happened to be at its end, so it announced the container was exhausted with
// 20,000px of list below. agent/loop.ts turns that token into "stop".
//
// The scroller is now resolved ONCE into a handle, and dispatch and both
// readings all use it — the target's survival stops mattering.
test('a list that recreates its rows still measures its own scroller', async () => {
  const RECREATE = `<!doctype html><title>recreate</title><h1>Feed</h1>
<div id="list" style="height:240px;overflow-y:auto;position:relative">
  <div id="spacer" style="height:20000px;position:absolute;top:0;left:0;width:1px"></div>
  <div id="rows" style="position:sticky;top:0"></div>
</div>
<script>
  const list = document.getElementById('list'), rows = document.getElementById('rows');
  const ROW = 20, N = 12;
  function render() {
    const first = Math.floor(list.scrollTop / ROW);
    rows.textContent = '';                       // destroy, then recreate
    for (let k = 0; k < N; k++) {
      const d = document.createElement('div');
      d.textContent = 'row ' + (first + k);
      rows.appendChild(d);
    }
  }
  list.addEventListener('scroll', render); render();
</script>`;
  const session = await start(RECREATE);
  try {
    const found = await session.dispatch({ verb: 'find', name: 'row 0' });
    const ref = ((found['matches'] as { ref: string }[]) ?? [])[0]?.ref;
    assert.ok(ref, JSON.stringify(found));

    const acted = await session.dispatch({ verb: 'act', ref, action: 'scroll' });
    const effect = effectOf(acted);

    // The delta must describe the LIST on both sides. The page is 900px wide in
    // this fixture and the list's extent is 20000 — so the extent is the tell.
    assert.match(effect.delta.before, /extent=20000/, effect.delta.before);
    assert.match(effect.delta.after, /extent=20000/,
      `after must measure the same scroller, not the page: ${effect.delta.after}`);
    assert.ok(!/at the end of this container/.test(effect.delta.after),
      `the list has 20,000px below; claiming exhaustion stops the model: ${effect.delta.after}`);

    // And the scroll must have actually moved the list, with new rows reachable.
    assert.equal(effect.verdict, 'verified', JSON.stringify(acted));
    assert.equal(effect.evidence, 'scrolled', JSON.stringify(effect));
    const later = await session.dispatch({ verb: 'find', name: 'row 12' });
    assert.ok(((later['matches'] as unknown[]) ?? []).length > 0,
      'rows past the first screen must be reachable after the scroll');
  } finally { await session.close(); }
});
