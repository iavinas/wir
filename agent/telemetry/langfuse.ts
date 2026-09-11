// The only file in the repo that names Langfuse. Delete it and the agent still
// compiles: `telemetry/index.ts` catches the failed dynamic import and returns the
// no-op. Everything is kept in ONE file on purpose — a split would need an internal
// import, and tsc elides `import { x } from './y.js'` when x is only a value, which
// silently turns every span into a no-op with a zero trace id (verified, v5.10).
//
// SDK: Langfuse v5 is OpenTelemetry-native; there is no `.trace()` client any more.
// A bare NodeTracerProvider + LangfuseSpanProcessor is the whole setup — no
// @opentelemetry/sdk-node, no auto-instrumentation.

import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { resourceFromAttributes } from '@opentelemetry/resources';
import { LangfuseSpanProcessor } from '@langfuse/otel';
import { startObservation, propagateAttributes } from '@langfuse/tracing';
import type { SpanContext } from '@opentelemetry/api';
import type {
  EpisodeMeta, EpisodeResultRecord, EpisodeTrace, GenerationRecord,
  Telemetry, ToolCallRecord,
} from './index.js';

export interface LangfuseOptions {
  publicKey: string;
  secretKey: string;
  baseUrl?: string | undefined;
}

// Trace-level metadata is Record<string,string> with a 200-char cap per value in
// v5 — anything richer belongs on an observation, where metadata is unrestricted.
const traceMeta = (v: unknown): string => {
  const s = typeof v === 'string' ? v : JSON.stringify(v) ?? '';
  return s.length > 200 ? `${s.slice(0, 197)}...` : s;
};

export function createLangfuseTelemetry(opts: LangfuseOptions): Telemetry {
  const processor = new LangfuseSpanProcessor({
    publicKey: opts.publicKey,
    secretKey: opts.secretKey,
    ...(opts.baseUrl ? { baseUrl: opts.baseUrl } : {}),
    environment: process.env['WIR_TELEMETRY_ENV'] ?? 'benchmark',
  });
  // register() must run before anything creates spans, and before any Langfuse
  // provider selection — attempt 6 lost a commit to this ordering: without it,
  // sibling observations become separate traces instead of children.
  const provider = new NodeTracerProvider({
    spanProcessors: [processor],
    // without this every trace reads service.name: unknown_service:node
    resource: resourceFromAttributes({ 'service.name': 'wir-agent' }),
  });
  provider.register();

  return {
    enabled: true,

    async run<T>(meta: EpisodeMeta, fn: (trace: EpisodeTrace) => Promise<T>): Promise<T> {
      // propagateAttributes is a scope: sessionId/tags/metadata reach every span
      // created inside it. Session = runId, so one attempt directory is one session.
      return propagateAttributes({
        traceName: `task-${meta.taskId}`,
        sessionId: meta.runId,
        userId: 'wir-agent',
        tags: [...(meta.benchmark ? [meta.benchmark] : []),
          meta.expectedAction, meta.model,
          ...(meta.suite ? [meta.suite] : [])],
        metadata: {
          runId: traceMeta(meta.runId),
          taskId: traceMeta(meta.taskId),
          expectedAction: traceMeta(meta.expectedAction),
          model: traceMeta(meta.model),
          ...(meta.benchmark ? { benchmark: traceMeta(meta.benchmark) } : {}),
          ...(meta.suite ? { suite: traceMeta(meta.suite) } : {}),
        },
      }, async () => {
        const root = startObservation('episode', {
          input: { taskId: meta.taskId, instruction: meta.instruction },
        }, { asType: 'agent' });

        // The method form of startObservation takes only { asType }; the module
        // form takes startTime + parentSpanContext, which is what we need to
        // record true durations. Parent explicitly by the root's span context.
        const rootCtx = (root as { otelSpan?: { spanContext?: () => SpanContext } })
          .otelSpan?.spanContext?.();
        const traceId: string | null = rootCtx?.traceId ?? null;

        const trace: EpisodeTrace = {
          traceId,
          generation(r: GenerationRecord): void {
            // Backdate the start by the measured duration: creating and ending a
            // span in the same instant reports 0 ms and destroys the latency
            // signal — the main reason to have telemetry at all.
            const endedAt = Date.now();
            const gen = startObservation('model', {
              model: r.model,
              input: r.input,          // the exact bytes we sent
            }, { asType: 'generation', startTime: new Date(endedAt - r.ms),
                 ...(rootCtx ? { parentSpanContext: rootCtx } : {}) });
            gen.update({
              output: r.output,
              usageDetails: { input: r.usage.input, output: r.usage.output,
                total: r.usage.input + r.usage.output },
              ...(r.error ? { level: 'ERROR', statusMessage: r.error } : {}),
            }).end(new Date(endedAt));
          },
          toolCall(r: ToolCallRecord): void {
            const endedAt = Date.now();
            const tool = startObservation(`verb:${r.verb}`, {
              input: r.args,
            }, { asType: 'tool', startTime: new Date(endedAt - r.ms),
                 ...(rootCtx ? { parentSpanContext: rootCtx } : {}) });
            tool.update({
              output: r.result,
              ...(r.error ? { level: 'ERROR', statusMessage: r.error } : {}),
            }).end(new Date(endedAt));
          },
          end(result: EpisodeResultRecord): void {
            root.update({
              output: {
                status: result.status,
                ...(result.answer !== undefined ? { answer: result.answer } : {}),
                modelCalls: result.modelCalls, wirCalls: result.wirCalls,
              },
              ...(result.status !== 'success'
                ? { level: 'WARNING', statusMessage: result.status } : {}),
            });
            root.end();   // the root ends LAST, after every child
          },
        };
        return fn(trace);
      });
    },

    async shutdown(): Promise<void> {
      // forceFlush is the only thing that ships spans; a process that exits without
      // it sends zero bytes. shutdown() throws ECONNREFUSED when the endpoint is
      // unreachable, so both are guarded — a telemetry outage must never fail a run
      // (index.ts::isolate also wraps this, belt and braces).
      try { await processor.forceFlush(); } catch { /* ignore */ }
      try { await processor.shutdown(); } catch { /* ignore */ }
    },
  };
}
