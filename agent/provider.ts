// OpenAI-compatible chat-completions client. Plain fetch, no dependencies.
// Config from env: WIR_PROVIDER_BASE_URL, WIR_PROVIDER_API_KEY, WIR_MODEL.

export interface ToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

/** A user message's content may be a typed part array rather than a bare string —
 *  the only shape that can carry an image. Built by code, never by the model: the
 *  model's whole output surface is tool_calls against a closed schema, and no verb
 *  accepts or returns an image. Same shape the OpenAI docs specify and browser-use
 *  builds (browser_use/agent/prompts.py, get_user_message). */
export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string; detail?: 'auto' | 'low' | 'high' } };

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | ContentPart[] | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export type { ToolDefinition } from '@wir/core';
import type { ToolDefinition } from '@wir/core';

export interface ChatResult {
  /** the exact request we sent — never a re-serialization (telemetry seam) */
  requestBody: unknown;
  ms: number;
  message: ChatMessage;
  inputTokens: number;
  outputTokens: number;
  /** null when the provider omitted the field — unknown is not zero */
  cachedTokens: number | null;
  // THE OUTPUT-SIDE BLIND SPOT. mimo-v2.5 is a reasoning model, and `outputTokens`
  // (= completion_tokens) INCLUDES its reasoning. Measured over 3,656 MACHINE-driven
  // calls: 708,262 output tokens billed, and only 35,725 (5.0%) appear anywhere in
  // the artifacts as content or tool-call arguments. 672,537 tokens — 95.0% — were
  // paid for, waited on, and never written down. Worst single call: 6,824 output
  // tokens producing 234 characters of content and a 5-character tool call, in 75
  // seconds. Verified live: "12 listed, 10 shown, how many hidden" returned 119
  // output tokens of which 115 were reasoning, for the one-character answer "2".
  //
  // MACHINE-driven is load-bearing. The first pass at this pooled all 5,211 recorded
  // calls, 1,555 of which are human-* studies where a PERSON drove the verbs by hand
  // — content null on all 1,555, median 10 output tokens, median 7,842 ms of human
  // think-time. That was 30% of calls and 50% of the fitted clock, and it inflated
  // every latency figure by ~50% while collapsing the fit's R^2 from 0.60 to 0.16.
  // The token ratio barely moved (94.8% -> 95.0%), but nothing else survived.
  //
  // That is the same class of blindness WIR_LOG_PROMPT exists to prevent on the
  // INPUT side, and it is the larger one: machine-only the clock is 56.3% fixed
  // per-round-trip / 43.7% token generation (ms = 2,988 + 12.0 * out_tok, R^2 0.60),
  // so the unobserved 95% of output tokens is most of the generation half.
  //
  // null, never 0, when the provider omits the field: a model that does not reason
  // and a field we failed to read must not look identical. Recorded for the debug
  // plane only — reasoningText is NEVER fed back to the model. Returning a
  // provider's reasoning into the next turn is a documented mistake on several
  // reasoning APIs, and doing it here would also break the append-only transcript
  // that keeps input 93.6% prefix-cached.
  reasoningTokens: number | null;
  /** the reasoning itself when the provider returns it, for the debug plane */
  reasoningText: string | null;
}

export class ProviderError extends Error {}

const DEFAULT_BASE_URL = 'https://api.openai.com/v1';
const DEFAULT_MODEL = 'gpt-4o-mini';

function envOr(name: string, fallback: string): string {
  const value = process.env[name];
  if (value !== undefined && value !== '') return value;
  process.stderr.write(`[provider] ${name} not set; defaulting to ${fallback}\n`);
  return fallback;
}

const baseUrl = envOr('WIR_PROVIDER_BASE_URL', DEFAULT_BASE_URL).replace(/\/+$/, '');
const model = envOr('WIR_MODEL', DEFAULT_MODEL);
export const modelName = model;
const providerTimeoutMs = Number(process.env['WIR_PROVIDER_TIMEOUT_MS'] ?? 120_000);
// Sampling was left at the provider's default until 2026-08-07, which made the
// instrument noisier than the effects measured with it: 7 of 30 re-runs of one
// task on one build flipped outcome (23%), so on a ten-task suite σ ≈ 1.3 tasks
// and every fix in the ledger sat under the noise floor. Both are configurable
// because a deliberate temperature sweep is a legitimate experiment — but the
// default is the one a score is quoted from, so the default is the pinned one.
//
// Measured against mimo-v2.5, 2026-08-07: both are real schema fields, not
// silently-dropped ones — temperature 400s outside [0, 1.5] and on a non-number,
// seed 400s on a non-integer, while a genuinely unknown key returns 200. Tool
// calling is unaffected at temperature 0. What this does NOT buy is a
// reproducible provider: repeated identical requests at temperature 0 with a
// fixed seed still returned different completions, so residual variance lives in
// the serving stack, above anything a client can set. This removes our own
// contribution to the noise; it does not make an episode replayable.
// THINKING, and why it is a flag rather than a default.
//
// mimo-v2.5 is a reasoning model and spends most of its output on reasoning
// tokens the runtime never sees: 14,462 of 16,289 (89%) on one stress-test
// episode, which at ~10.6 ms/token is 173 s of the 218 s that episode took.
// Batching removes the FIXED per-round-trip cost and cannot touch that half.
//
// The endpoint honours a disable, verified rather than assumed — the same
// probe discipline temperature and seed got, because provider.ts already
// records that an unknown key here returns 200 and is silently dropped:
//   baseline                       completion=181  reasoning=177  content "2"
//   thinking:{type:"disabled"}     completion=2    reasoning=0    content "2"
// Same answer, 90x fewer tokens.
//
// OFF BY DEFAULT. This changes how the model thinks, not merely how it is
// billed, and a score quoted from a thinking-disabled arm is a different
// measurement — it must be declared, not inherited. `make flags` prints it.
export const thinkingDisabled = process.env['WIR_DISABLE_THINKING'] === '1';
export const temperature = Number(process.env['WIR_TEMPERATURE'] ?? 0);
export const seed = Number(process.env['WIR_SEED'] ?? 20260807);
const apiKey = process.env['WIR_PROVIDER_API_KEY'] ?? '';
if (apiKey === '') {
  process.stderr.write('[provider] WIR_PROVIDER_API_KEY not set; requests carry no Authorization header\n');
}

interface CompletionPayload {
  choices?: { message?: ChatMessage & {
    // Reasoning models return their thinking beside `content`, under a key the
    // OpenAI-compatible schema never standardised: `reasoning_content` (DeepSeek
    // and the providers that copied it, mimo included) or `reasoning`. Read both,
    // prefer neither — whichever is a non-empty string wins.
    reasoning_content?: unknown; reasoning?: unknown;
  } }[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    prompt_tokens_details?: { cached_tokens?: number };
    completion_tokens_details?: { reasoning_tokens?: number };
  };
}

async function postWithRetry(body: string): Promise<CompletionPayload> {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (apiKey !== '') headers['authorization'] = `Bearer ${apiKey}`;
  let lastFailure = '';
  const backoffMs = [0, 2000, 8000, 20000];
  for (let attempt = 0; attempt < 4; attempt += 1) {
    if (attempt > 0) {
      process.stderr.write(`[provider] retry ${attempt} after: ${lastFailure}\n`);
      await new Promise(r => setTimeout(r, backoffMs[attempt] ?? 20000));
    }
    try {
      // A hung provider connection must not outlive the episode: without an
      // abort signal the wall-time budget can never fire, because the loop that
      // checks it is blocked inside fetch. The bound is configurable because a
      // provider's honest think-time is not ours to assume — a reasoning model
      // behind a slow queue, or a human driving the verbs by hand
      // (docs/plans/human-in-the-loop-spec.md), both need longer than two minutes.
      const response = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST', headers, body, signal: AbortSignal.timeout(providerTimeoutMs),
      });
      // 429/408 are transient like 5xx (review A3: one rate-limit blip ended
      // the episode on attempt 0 — and a whole suite of them reads as broken).
      if (response.status < 500 && response.status !== 429 && response.status !== 408) {
        if (!response.ok) {
          const detail = (await response.text()).slice(0, 300);
          throw new ProviderError(`provider returned ${response.status}: ${detail}`);
        }
        // The body is read INSIDE the loop because the abort signal stays armed
        // until the last byte arrives: a timeout firing mid-body used to escape
        // to the caller's parse and come back terminal as "unparseable JSON" —
        // measured at 219 discarded model calls over 6 episodes with zero retry
        // lines (docs/plans/fewer-misses.md §B3). text() throwing here is
        // transport, retried below like any other; JSON.parse failing on a body
        // we fully received is the provider's failure class, not ours (review:
        // it surfaced as agent_failed and polluted the taxonomy) — terminal.
        const text = await response.text();
        try {
          return JSON.parse(text) as CompletionPayload;
        } catch (e) {
          throw new ProviderError(`provider returned unparseable JSON: ${String(e).slice(0, 120)}`);
        }
      }
      lastFailure = `status ${response.status}`;
    } catch (error) {
      if (error instanceof ProviderError) throw error;
      lastFailure = error instanceof Error ? error.message : String(error);
    }
  }
  throw new ProviderError(`provider unreachable after retry: ${lastFailure}`);
}

// A truncated base64 image is the failure mode that costs a whole episode and
// leaves no trace: the model receives a corrupt picture, behaves oddly, and every
// number downstream is quietly wrong. Nothing in the send path truncates today,
// but the observability paths around it are written to be length-safe, and
// log-then-send on one object is how a clipped string becomes what ships.
//
// So the guarantee is positive, not an audit: an image is intact iff it still
// carries its own terminator. JPEG ends with FFD9, PNG with the IEND chunk — a
// payload cut anywhere loses it. This needs no side channel and no expected
// length, and it makes silent truncation impossible rather than unlikely.
const TERMINATORS: Record<string, Buffer> = {
  jpeg: Buffer.from([0xff, 0xd9]),
  png: Buffer.from([0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82]),
};

function assertImagePartsIntact(messages: ChatMessage[]): void {
  messages.forEach((message, index) => {
    if (!Array.isArray(message.content)) return;
    for (const part of message.content) {
      if (part.type !== 'image_url') continue;
      const match = /^data:image\/(jpeg|png);base64,(.*)$/s.exec(part.image_url.url);
      if (match === null) {
        throw new ProviderError(
          `message ${index}: image part is not a jpeg/png base64 data URI`);
      }
      const [, format, payload] = match as unknown as [string, string, string];
      if (payload.length % 4 !== 0) {
        throw new ProviderError(
          `message ${index}: base64 image payload truncated (length ${payload.length} is not a multiple of 4)`);
      }
      const bytes = Buffer.from(payload, 'base64');
      const terminator = TERMINATORS[format] as Buffer;
      if (!bytes.subarray(-terminator.length).equals(terminator)) {
        throw new ProviderError(
          `message ${index}: ${format} image truncated — ${bytes.length} bytes decoded, terminator missing`);
      }
    }
  });
}

/** The provider's own reasoning text, or null. Never fed back to the model. */
function reasoningOf(message: unknown): string | null {
  const m = message as { reasoning_content?: unknown; reasoning?: unknown };
  for (const v of [m?.reasoning_content, m?.reasoning]) {
    if (typeof v === 'string' && v !== '') return v;
  }
  return null;
}

export async function chatComplete(
  messages: ChatMessage[],
  tools: ToolDefinition[],
  // Pass-through, default unchanged: 'auto' for the episode loop, a forced
  // single function where the caller needs the reply to BE a tool call (the
  // finish evaluator — its JSON-in-text transport measured broken on mimo,
  // 4 of 7 verdicts unparseable; see agent/loop.ts VERDICT_TOOL).
  toolChoice: 'auto' | { type: 'function'; function: { name: string } } = 'auto',
): Promise<ChatResult> {
  assertImagePartsIntact(messages);
  // BATCHING IS THE WALL-CLOCK LEVER. 96% of an episode's clock is model latency at
  // ~5.1s median per round trip, against a 41ms median WIR dispatch — so a saved
  // ROUND TRIP is ~5 seconds and a saved millisecond inside the runtime is nothing.
  //
  // Measured before asking for it: 11,588 of 11,730 recorded model turns (98.8%)
  // emitted exactly ONE tool call. The loop has always executed batches
  // (agent/loop.ts, `for (const call of toolCalls)`); nothing ever requested them.
  // A 22-call episode at one verb per turn is 112s; the same verbs at ~3 per turn
  // is 41s.
  //
  // PROBED 2026-08-19 against mimo-v2.5: `parallel_tool_calls` is INERT here.
  // Three independent tools, one prompt that plainly warrants all three:
  //   parallel_tool_calls: true  -> 3 tool_calls
  //   parallel_tool_calls: false -> 3 tool_calls   (would be 1 if honoured)
  //   field omitted              -> 3 tool_calls
  //   a deliberately bogus key   -> 3 tool_calls, HTTP 200  (the control: this
  //                                 endpoint drops unknown keys silently)
  // So the flag buys nothing and costs nothing; it stays for other providers.
  // The load-bearing half of that probe is the flip side: mimo emitted 3 of 3
  // when the instruction was unmistakable. It CAN batch. On real episodes it
  // does so on ~4-5% of turns while 58.3% end with the page unmoved, so the
  // lever is where the instruction sits (tool schema, turn-local feedback),
  // never this field.
  const requestBody = { model, messages, tools, tool_choice: toolChoice,
    parallel_tool_calls: true, temperature, seed,
    ...(thinkingDisabled ? { thinking: { type: 'disabled' } } : {}) };
  const body = JSON.stringify(requestBody);
  const startedAt = Date.now();
  const payload = await postWithRetry(body);
  const message = payload.choices?.[0]?.message;
  if (message === undefined) {
    throw new ProviderError(`provider response had no message: ${JSON.stringify(payload).slice(0, 300)}`);
  }
  return {
    requestBody,
    ms: Date.now() - startedAt,
    message,
    inputTokens: payload.usage?.prompt_tokens ?? 0,
    outputTokens: payload.usage?.completion_tokens ?? 0,
    cachedTokens: payload.usage?.prompt_tokens_details?.cached_tokens ?? null,
    reasoningTokens: payload.usage?.completion_tokens_details?.reasoning_tokens ?? null,
    reasoningText: reasoningOf(message),
  };
}

// Tool definitions live in core/toolschemas.ts — one source of truth with the
// wire contract; re-exported here for the loop.
export { toolDefinitions, toolDefinitionsFor } from '@wir/core';
