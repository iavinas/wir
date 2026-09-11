// Credential posture: a value filled into a page is the one thing WIR handles that
// can be a password, and telemetry ships to a third party. Redaction is STRUCTURAL —
// it fires on the position a fill value occupies in a known shape, never on a
// pattern that "looks like" a secret — and it happens at the telemetry boundary
// only. `trajectory.jsonl` stays verbatim: it is the local lossless debug plane,
// and the one place a garbled fill is diagnosable.
//
// Partial redaction would be worse than none, because it reads as protection.
// A fill value reaches the boundary in SIX positions, and all six are covered:
//   1. the act tool span's input                     (args.value)
//   2. the act tool span's output — the effect delta (act.ts embeds the value in
//      `value="…"`, identified by evidence value_set / value_mismatch)
//   3. the next generation's input transcript, twice over: as the assistant's own
//      tool_calls[].function.arguments, and as the tool-role message carrying the
//      act result from position 2
//   4. the generation's output — the assistant message that ORIGINATES the value
//   5. EVERY LATER `read` of that page — read.ts projects a node's current `value`
//      on both controls and children, so a filled password is re-emitted for the
//      rest of the episode
//   6. the EPISODE RESULT — `end({answer})` ships the model's final answer to the
//      vendor, and the system prompt tells the model to report a field's value as
//      "the page's verbatim string, units and all", so an answer can echo what was
//      typed
// Positions 4, 5 and 6 are not in the step's list of three. Leaving them out would
// have leaked every value on its first appearance (4), on every subsequent read (5),
// and in the final answer (6) — exactly the silent-partial failure this file warns
// about. Positions 5 and 6 were each found by adversarial review, not by design;
// that is twice this claim has been wrong, which is why the count is now stated as
// a claim the tests must keep true rather than as a comment.
//
// Position 6 cannot be keyed on STRUCTURE — an answer is model-authored prose, and
// nothing about its shape says "secret". It is closed instead by MEMORY: the
// redactor already sees every fill, so it remembers the exact values this episode
// typed and rewrites those exact strings if they reappear in the answer. That is an
// exact match against a known secret, not a pattern guess at what looks like one.
//
// Opt out with WIR_TELEMETRY_LOG_VALUES=1.

import type { GenerationRecord, ToolCallRecord } from './index.js';

const VALUE_BEARING_EVIDENCE = new Set(['value_set', 'value_mismatch']);

function marker(value: string): string {
  return `[redacted:len=${value.length}]`;
}

export function valueRedactionEnabled(): boolean {
  return process.env['WIR_TELEMETRY_LOG_VALUES'] !== '1';
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

/** `{verb:'act', action:'fill', value:'…'}` — the request shape, wherever it appears. */
function redactFillArgs(args: unknown): unknown {
  if (!isRecord(args)) return args;
  if (args['action'] !== 'fill' || typeof args['value'] !== 'string') return args;
  return { ...args, value: marker(args['value']) };
}

/** act.ts writes the value into the delta as `value="…"` (JSON-encoded). */
function redactDeltaSide(side: unknown): unknown {
  if (typeof side !== 'string') return side;
  return side.replace(/^value=("(?:[^"\\]|\\.)*")/, (_m, json: string) => {
    try { return `value=${marker(String(JSON.parse(json)))}`; } catch { return 'value=[redacted]'; }
  });
}

/** A projected node's current value, wherever it appears in a read result. The
 *  test is structural: an object that carries BOTH a `ref` and a string `value` is
 *  a projected node (read.ts controlSummary / nodeDetail), and telemetry has no
 *  need of the literal contents of a form field. */
function redactProjectedValues(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(redactProjectedValues);
  if (!isRecord(node)) return node;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(node)) out[k] = redactProjectedValues(v);
  if (typeof out['ref'] === 'string' && typeof out['value'] === 'string') {
    out['value'] = marker(out['value'] as string);
  }
  return out;
}

/** An act result whose evidence says a value was read back. */
function redactActResult(result: unknown): unknown {
  if (!isRecord(result)) return result;
  const effect = result['effect'];
  if (!isRecord(effect)) return result;
  if (typeof effect['evidence'] !== 'string' || !VALUE_BEARING_EVIDENCE.has(effect['evidence'])) {
    return result;
  }
  const delta = effect['delta'];
  if (!isRecord(delta)) return result;
  return {
    ...result,
    effect: {
      ...effect,
      delta: { ...delta, before: redactDeltaSide(delta['before']), after: redactDeltaSide(delta['after']) },
    },
  };
}

// A screenshot data URI is ~250KB of base64 per call. It is stripped from the
// telemetry boundary UNCONDITIONALLY — not under valueRedactionEnabled, which is a
// credential opt-out and has nothing to say about size. Three reasons it can never
// be useful here: the vendor rejects or silently clips payloads that size (a clipped
// transcript is a lying transcript), it would bury every readable field in the span,
// and the bytes already exist on disk under the attempt's screenshots/ directory —
// which is the copy a human can actually open.
export function stripImageParts(content: unknown): unknown {
  if (!Array.isArray(content)) return content;
  return content.map(part => {
    if (!isRecord(part) || part['type'] !== 'image_url') return part;
    const url = isRecord(part['image_url']) ? part['image_url']['url'] : undefined;
    if (typeof url !== 'string') return part;
    const format = /^data:image\/([a-z]+);base64,/.exec(url)?.[1] ?? 'unknown';
    const payloadLength = Math.max(0, url.length - url.indexOf(',') - 1);
    const image = part['image_url'] as Record<string, unknown>;
    return { ...part, image_url: { ...image, url: `[image:${format} b64len=${payloadLength}]` } };
  });
}

/** An assistant message's tool_calls carry the arguments as a JSON STRING. */
function redactMessage(message: unknown): unknown {
  if (!isRecord(message)) return message;
  let out = message;

  // A user message's content may be a typed part array carrying an image.
  const stripped = stripImageParts(message['content']);
  if (stripped !== message['content']) out = { ...out, content: stripped };

  const calls = message['tool_calls'];
  if (Array.isArray(calls)) {
    out = { ...out, tool_calls: calls.map(call => {
      if (!isRecord(call)) return call;
      const fn = call['function'];
      if (!isRecord(fn) || fn['name'] !== 'act' || typeof fn['arguments'] !== 'string') return call;
      let parsed: unknown;
      try { parsed = JSON.parse(fn['arguments']); } catch { return call; }
      const redacted = redactFillArgs(parsed);
      if (redacted === parsed) return call;
      return { ...call, function: { ...fn, arguments: JSON.stringify(redacted) } };
    }) };
  }

  // A tool-role message's content is the stringified verb result.
  if (message['role'] === 'tool' && typeof message['content'] === 'string') {
    try {
      const parsed = JSON.parse(message['content']) as unknown;
      const redacted = redactProjectedValues(redactActResult(parsed));
      if (JSON.stringify(redacted) !== message['content']) {
        out = { ...out, content: JSON.stringify(redacted) };
      }
    } catch { /* not JSON: nothing structural to redact */ }
  }
  return out;
}

/** Values this episode typed into the page. Remembered so position 6 can be closed
 *  by exact match; scoped per episode by createRedactor(). */
export interface Redactor {
  toolCall(record: ToolCallRecord): ToolCallRecord;
  generation(record: GenerationRecord): GenerationRecord;
  episodeAnswer(answer: string | undefined): string | undefined;
}

export function createRedactor(): Redactor {
  const filled = new Set<string>();
  const remember = (args: unknown): void => {
    if (isRecord(args) && args['action'] === 'fill' && typeof args['value'] === 'string'
        && args['value'] !== '') {
      filled.add(args['value']);
    }
  };
  return {
    toolCall(record) {
      if (!valueRedactionEnabled()) return record;
      remember(record.args);
      return {
        ...record,
        args: redactFillArgs(record.args),
        result: redactProjectedValues(redactActResult(record.result)),
      };
    },
    generation(record) {
      // No opt-out guard here: redactGenerationRecord strips images either way.
      return redactGenerationRecord(record);
    },
    episodeAnswer(answer) {
      if (!valueRedactionEnabled() || answer === undefined) return answer;
      let out = answer;
      for (const secret of filled) out = out.split(secret).join(marker(secret));
      return out;
    },
  };
}

export function redactToolCallRecord(record: ToolCallRecord): ToolCallRecord {
  if (!valueRedactionEnabled()) return record;
  return {
    ...record,
    args: redactFillArgs(record.args),
    result: redactProjectedValues(redactActResult(record.result)),
  };
}

/** Images go regardless of the credential opt-out; values go unless opted out. */
function stripImagesOnly(message: unknown): unknown {
  if (!isRecord(message)) return message;
  const stripped = stripImageParts(message['content']);
  return stripped === message['content'] ? message : { ...message, content: stripped };
}

export function redactGenerationRecord(record: GenerationRecord): GenerationRecord {
  const scrub = valueRedactionEnabled() ? redactMessage : stripImagesOnly;
  let input = record.input;
  if (isRecord(input) && Array.isArray(input['messages'])) {
    input = { ...input, messages: input['messages'].map(scrub) };
  }
  return { ...record, input, output: scrub(record.output) };
}
