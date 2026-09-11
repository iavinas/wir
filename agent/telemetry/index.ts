// Telemetry seam. Vendor-neutral by construction: this file names no product.
// Removing the integration is `rm langfuse.ts` plus one clean build — a stale
// dist/ copy keeps loading until then. The import below keeps its specifier out
// of the typechecker's reach so the deletion survives `npm run build` (proven
// 2026-08-04: a literal specifier broke the rebuild, falsifying the original
// one-file-removability claim; clean-build runtime falls back to the no-op).
// Disabling it is unsetting the credentials, or WIR_TELEMETRY=off.
//
// Rule that outranks every feature here: telemetry may never fail an episode.
// A tracing outage must not become a benchmark zero

import { createRedactor } from './redact.js';

export interface EpisodeMeta {
  runId: string;
  taskId: number;
  instruction: string;
  expectedAction: string;
  model: string;
  benchmark?: string;
  suite?: string;
}

export interface GenerationRecord {
  model: string;
  input: unknown;         // the exact request body we sent, never a re-serialization
  output: unknown;        // the exact message we got back
  usage: { input: number; output: number };
  ms: number;
  error?: string;
}

export interface ToolCallRecord {
  verb: string;
  args: unknown;
  result: unknown;
  ms: number;
  error?: string;
}

export interface EpisodeResultRecord {
  status: string;
  answer?: string;
  modelCalls: number;
  wirCalls: number;
}

export interface EpisodeTrace {
  /** Provider-side trace id, when the backend exposes one. Written into the
   *  attempt directory so artifact → trace lookup works, not just trace → artifact. */
  readonly traceId: string | null;
  generation(record: GenerationRecord): void;
  toolCall(record: ToolCallRecord): void;
  end(result: EpisodeResultRecord): void;
}

export interface Telemetry {
  readonly enabled: boolean;
  /** Runs the episode inside the trace's scope. Scope-shaped because trace-level
   *  attributes (session, tags) only propagate to spans created within it. */
  run<T>(meta: EpisodeMeta, fn: (trace: EpisodeTrace) => Promise<T>): Promise<T>;
  shutdown(): Promise<void>;
}

const noopTrace: EpisodeTrace = {
  traceId: null,
  generation() { /* no-op */ },
  toolCall() { /* no-op */ },
  end() { /* no-op */ },
};

export const noopTelemetry: Telemetry = {
  enabled: false,
  run: async (_meta, fn) => fn(noopTrace),
  shutdown: async () => undefined,
};

/** Wraps a Telemetry so no call can ever throw into the episode — and so no fill
 *  value leaves the machine through it. Redaction lives HERE, at the
 *  vendor-neutral seam, not in the backend file: every record already passes
 *  through this guard, so a second backend cannot forget to do it. */
export function isolate(inner: Telemetry): Telemetry {
  const warn = (where: string, e: unknown): void => {
    process.stderr.write(`[telemetry] ${where} failed (ignored): ${String(e)}\n`);
  };
  return {
    enabled: inner.enabled,
    async run(meta, fn) {
      // One redactor per episode: it remembers the values THIS episode typed so the
      // final answer can be cleaned by exact match (position 6).
      const redactor = createRedactor();
      const guard = (t: EpisodeTrace): EpisodeTrace => ({
        get traceId() { try { return t.traceId; } catch { return null; } },
        generation(r) {
          try { t.generation(redactor.generation(r)); } catch (e) { warn('generation', e); }
        },
        toolCall(r) {
          try { t.toolCall(redactor.toolCall(r)); } catch (e) { warn('toolCall', e); }
        },
        end(r) {
          try {
            const answer = redactor.episodeAnswer(r.answer);
            t.end(answer === undefined ? r : { ...r, answer });
          } catch (e) { warn('end', e); }
        },
      });
      try {
        return await inner.run(meta, async (t) => {
          try {
            return await fn(guard(t));
          } catch (e) {
            // Tag episode errors so the outer catch can tell them from
            // telemetry failures. Without the tag (review finding: the guard
            // was inert — nothing ever set it), an episode error was treated
            // as a telemetry failure and fn RE-RAN with a second browser —
            // on a MUTATE task, mutating the site twice.
            if (e !== null && typeof e === 'object') {
              (e as { __wirEpisodeError?: boolean }).__wirEpisodeError = true;
            }
            throw e;
          }
        });
      } catch (e) {
        if ((e as { __wirEpisodeError?: boolean })?.__wirEpisodeError) throw e;
        warn('run', e);
        return fn(noopTrace);
      }
    },
    async shutdown() {
      try { await inner.shutdown(); } catch (e) { warn('shutdown', e); }
    },
  };
}

/**
 * The only place a vendor is named. The specifier is deliberately a variable:
 * a literal would be resolved by tsc, and `rm langfuse.ts` must leave a
 * compiling agent (the catch below turns the missing module into the no-op).
 */
export async function createTelemetry(): Promise<Telemetry> {
  if (process.env['WIR_TELEMETRY'] === 'off') return noopTelemetry;
  const pub = process.env['LANGFUSE_PUBLIC_KEY'];
  const sec = process.env['LANGFUSE_SECRET_KEY'];
  if (!pub || !sec) return noopTelemetry;
  try {
    const vendorModule = './langfuse.js';
    const { createLangfuseTelemetry } = await import(vendorModule) as {
      createLangfuseTelemetry: (opts: {
        publicKey: string; secretKey: string; baseUrl?: string | undefined;
      }) => Telemetry;
    };
    return isolate(createLangfuseTelemetry({
      publicKey: pub, secretKey: sec,
      baseUrl: process.env['LANGFUSE_BASE_URL'] ?? process.env['LANGFUSE_HOST'] ?? undefined,
    }));
  } catch (e) {
    process.stderr.write(`[telemetry] disabled — module unavailable: ${String(e)}\n`);
    return noopTelemetry;
  }
}
