// NEXT-CALL OFFERS: the literal `until` / `expect` call, built from the act the
// model just made, appended at the moment the runtime's own result shows the
// pattern those arguments exist to replace.
//
// Measured on develop-v2 arm 4 (20 episodes): `until` and `expect` — both
// merged, both in the tool schema, one prompt sentence each — were used zero
// times. The patterns they replace are in the same arm's trajectories:
// shopping_admin 464 re-scrolled a page that answered scrolled_no_new_content
// while waiting for a stage to render; gitlab 660 followed 20 of 35 acts with
// an immediate `read` to see whether the act had taken; shopping 521 submitted
// a form whose receipt carried a value the model did not mean. The house rule
// (CLAUDE.md, wir-cli): the same rules given as advice lose and given as the
// literal next call pass — and a continuation the runtime can compute is
// offered as that call, never as prose.
//
// AGENT LAYER, pure. No task id, no site word, no page text is read: every
// offer is assembled from the model's own arguments and the fields the act
// result already carried (evidence, delta, receipt). Nothing here decides
// anything — an offer is one line the model may copy, and a model that ignores
// it twice is not shown a third.

export type OfferPattern = 'until' | 'expect';

/** Every offer stays under this many bytes: a line the model can copy, never a
 *  second prompt. */
export const OFFER_MAX_BYTES = 200;
/** An offer the model has walked past this many times is retired for the
 *  episode. */
export const OFFER_IGNORE_LIMIT = 2;

interface SeenCall {
  verb: string;
  args: Record<string, unknown>;
  response: Record<string, unknown>;
}

export interface OfferState {
  lastCall: SeenCall | null;
  /** The previous ACT (rejections excluded) — the consecutive-scroll rule reads it. */
  lastAct: SeenCall | null;
  /** Targets already offered `until`; one offer per target, then silence. */
  untilOffered: Set<string>;
  /** `expect` shapes already offered this episode: value | checked | sent. */
  expectShapesOffered: Set<string>;
  ignored: Record<OfferPattern, number>;
  taken: Record<OfferPattern, number>;
  /** An offer whose fate the next act decides. */
  pending: OfferPattern | null;
}

export function createOfferState(): OfferState {
  return {
    lastCall: null, lastAct: null, untilOffered: new Set(), expectShapesOffered: new Set(),
    ignored: { until: 0, expect: 0 }, taken: { until: 0, expect: 0 }, pending: null,
  };
}

export interface Offer {
  pattern: OfferPattern;
  /** Which construction the line carries — for the record, never for the model. */
  shape: string;
  /** The line appended to the tool result, OFFER_MAX_BYTES or fewer. */
  line: string;
  ref: string;
}

/** What the last act taught about the offer before it: the model's next act
 *  either carried the offered key or did not. */
export interface OfferOutcome { pattern: OfferPattern; taken: boolean; ignored: number }

const str = (v: unknown): string | null => (typeof v === 'string' ? v : null);
const obj = (v: unknown): Record<string, unknown> | null =>
  (typeof v === 'object' && v !== null && !Array.isArray(v) ? v as Record<string, unknown> : null);

function effectOf(response: Record<string, unknown>): { verdict: string | null; evidence: string | null; after: string | null } {
  const effect = obj(response['effect']);
  const delta = effect === null ? null : obj(effect['delta']);
  return {
    verdict: effect === null ? null : str(effect['verdict']),
    evidence: effect === null ? null : str(effect['evidence']),
    after: delta === null ? null : str(delta['after']),
  };
}

function bytes(s: string): number { return Buffer.byteLength(s, 'utf8'); }

/** The call, with its arguments in the flat wire order, optionally trimmed to fit. */
function call(parts: Record<string, unknown>): string { return JSON.stringify(parts); }

/** An act's own arguments, minus anything the offer replaces. */
function baseArgs(args: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { verb: 'act', ref: args['ref'], action: args['action'] };
  if (typeof args['value'] === 'string') out['value'] = args['value'];
  return out;
}

/** Fit a line under the byte cap: first drop the alternative, then stand a
 *  marked placeholder in for a long value — an offer is a shape to copy, and a
 *  cut string would be a wrong literal rather than a shorter one. */
function fit(lead: string, args: Record<string, unknown>, tail: string, alt: string): string {
  const full = `${lead}${call({ ...args, ...JSON.parse(tail) as Record<string, unknown> })}${alt}`;
  if (bytes(full) <= OFFER_MAX_BYTES) return full;
  const noAlt = `${lead}${call({ ...args, ...JSON.parse(tail) as Record<string, unknown> })}`;
  if (bytes(noAlt) <= OFFER_MAX_BYTES) return noAlt;
  const shortArgs = { ...args, ...(typeof args['value'] === 'string' ? { value: '<the value>' } : {}) };
  const shortTail = (JSON.parse(tail) as Record<string, unknown>);
  const st = obj(shortTail['expect']);
  const state = st === null ? null : obj(st['state']);
  if (state !== null && typeof state['value'] === 'string') state['value'] = '<the value>';
  return `${lead}${call({ ...shortArgs, ...shortTail })}`;
}

/** Pattern 1 — an act the model is going to wait on. The second consecutive
 *  scroll on one target answered scrolled_no_new_content, or any act whose
 *  verdict is unknown / whose evidence is no_observable_change_yet. `network`
 *  is not offered on scroll: the runtime refuses it there (scroll arms no
 *  observers), and an offer the runtime would reject is not a continuation. */
function untilOffer(state: OfferState, args: Record<string, unknown>, response: Record<string, unknown>): Offer | null {
  const ref = str(args['ref']);
  const action = str(args['action']);
  if (ref === null || action === null) return null;
  if (args['until'] !== undefined && args['until'] !== null) return null;
  const { verdict, evidence } = effectOf(response);
  const prev = state.lastAct;
  const secondScroll = action === 'scroll' && evidence === 'scrolled_no_new_content'
    && prev !== null && str(prev.args['action']) === 'scroll' && str(prev.args['ref']) === ref
    && effectOf(prev.response).evidence === 'scrolled_no_new_content';
  const unmoved = verdict === 'unknown' || evidence === 'no_observable_change_yet';
  if (!secondScroll && !unmoved) return null;
  if (state.untilOffered.has(ref)) return null;
  const alt = action === 'scroll' ? ' or "until":{"gone":"<a loading label>"}' : ' or "until":{"network":"idle"}';
  const line = fit('Waiting for something? Say it on the act: ', baseArgs(args),
    '{"until":{"text":"<the words you expect>"}}', alt);
  return { pattern: 'until', shape: secondScroll ? 'scroll' : 'unmoved', line, ref };
}

/** Pattern 2 — a `read` straight after an act whose evidence was local-only:
 *  the model is checking whether the act took. The next act can declare that
 *  instead; the offer is that declaration, shaped from the act just made.
 *
 *  Any read that is not a pagination step, not only the overview. The brief
 *  named `read {}` from arm 4's gitlab 660 (fill, then read {}); the first
 *  episodes on this build checked their fills with `read {target: the field's
 *  region}` every time and the overview never — the same check, aimed. A
 *  cursor is the continuation of a read already in hand, not a check. */
function expectOffer(state: OfferState, readArgs: Record<string, unknown>): Offer | null {
  if (readArgs['cursor'] !== undefined && readArgs['cursor'] !== null) return null;
  const prev = state.lastCall;
  if (prev === null || prev.verb !== 'act' || prev.response['rejected'] !== undefined) return null;
  if (prev.args['expect'] !== undefined && prev.args['expect'] !== null) return null;
  const ref = str(prev.args['ref']);
  const action = str(prev.args['action']);
  if (ref === null || action === null) return null;
  const { evidence, after } = effectOf(prev.response);
  const value = str(prev.args['value']);
  let shape: string | null = null;
  let tail = '';
  let lead = 'Next act, declare what it should do: ';
  if ((evidence === 'value_set' || evidence === 'text_typed') && value !== null) {
    shape = 'value';
    tail = JSON.stringify({ expect: { state: { value } } });
  } else if (action === 'click' && evidence === 'target_state_changed' && after !== null && /checked=(true|false)/.test(after)) {
    shape = 'checked';
    tail = JSON.stringify({ expect: { state: { checked: /checked=true/.test(after) } } });
  } else if (action === 'click' && evidence === 'dom_mutated') {
    shape = 'sent';
    lead = 'On the click that submits, declare it: ';
    tail = JSON.stringify({ expect: { sent: { method: 'POST' }, text: '<words that should appear>' } });
  }
  if (shape === null || state.expectShapesOffered.has(shape)) return null;
  return { pattern: 'expect', shape, line: fit(lead, baseArgs(prev.args), tail, ''), ref };
}

/** Feed every WIR call, in order, as it returns. The result is what to append
 *  to THIS call's tool result (or nothing) and, when this call was an act, the
 *  fate of the offer before it. Pure over its state: no I/O. */
export function offerAfter(
  state: OfferState, verb: string, args: Record<string, unknown>, response: Record<string, unknown>,
): { offer: Offer | null; outcome: OfferOutcome | null } {
  let outcome: OfferOutcome | null = null;
  let offer: Offer | null = null;
  const rejected = response['rejected'] !== undefined;
  if (verb === 'act' && state.pending !== null) {
    const key = state.pending;
    const taken = args[key] !== undefined && args[key] !== null;
    if (taken) state.taken[key] += 1; else state.ignored[key] += 1;
    outcome = { pattern: key, taken, ignored: state.ignored[key] };
    state.pending = null;
  }
  if (verb === 'act' && !rejected && state.ignored.until < OFFER_IGNORE_LIMIT) {
    offer = untilOffer(state, args, response);
  } else if (verb === 'read' && !rejected && state.ignored.expect < OFFER_IGNORE_LIMIT) {
    offer = expectOffer(state, args);
  }
  if (offer !== null) {
    state.pending = offer.pattern;
    if (offer.pattern === 'until') state.untilOffered.add(offer.ref);
    else state.expectShapesOffered.add(offer.shape);
  }
  if (verb === 'act' && !rejected) state.lastAct = { verb, args, response };
  state.lastCall = { verb, args, response };
  return { offer, outcome };
}

/** Pattern 3 — the MUTATE confrontation. For a cited act that declared no
 *  expectation, the literal `expect.sent` call it would have taken, built from
 *  its own receipt: the first non-GET request's method and path, plus the body
 *  fields whose values the model itself filled this episode. null when the
 *  receipt holds no non-GET request — there is nothing to build from, and the
 *  caller says so instead. */
export function sentExpectCall(
  args: Record<string, unknown>, response: Record<string, unknown>, filled: ReadonlySet<string>,
): string | null {
  const receipt = obj(response['receipt']);
  const requests = Array.isArray(receipt?.['requests']) ? receipt['requests'] as unknown[] : [];
  const first = requests.map(obj).find(r => r !== null && typeof r['method'] === 'string' && r['method'] !== 'GET');
  if (first === null || first === undefined) return null;
  const method = String(first['method']);
  let path = String(first['url'] ?? '');
  try { path = new URL(path).pathname; } catch { /* keep as given */ }
  const body = obj(first['body']);
  const fields = body === null ? null : obj(body['fields']);
  const mine: Record<string, string> = {};
  for (const [k, v] of Object.entries(fields ?? {})) {
    if (typeof v === 'string' && v.trim() !== '' && filled.has(v.trim())) mine[k] = v;
  }
  const base = { verb: 'act', ref: args['ref'], action: args['action'] };
  const keys = Object.keys(mine);
  // Fields drop from the end until the line fits; the path is never cut.
  for (let n = keys.length; n >= 0; n -= 1) {
    const kept = Object.fromEntries(keys.slice(0, n).map(k => [k, mine[k]]));
    const sent: Record<string, unknown> = { method, path };
    if (n > 0) sent['fields'] = kept;
    const line = call({ ...base, expect: { sent } });
    if (bytes(line) <= OFFER_MAX_BYTES) return line;
  }
  return call({ ...base, expect: { sent: { method } } });
}
