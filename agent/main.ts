// Entrypoint: one JSON request on stdin, one JSON response on stdout, logs on stderr.
// Exits 0 whenever the one-response contract was honored, even if the task failed.

import { runEpisode, type AgentRequest, type AgentStatus, type Metrics } from './loop.js';
import { temperature, seed, modelName } from './provider.js';

const PROTOCOL_VERSION = 'wa-benchmark-agent-1';

interface AgentResponse {
  protocolVersion: typeof PROTOCOL_VERSION;
  runId: string;
  status: AgentStatus;
  finalResponse: Record<string, unknown>;
  metrics: Metrics;
  effectiveBrowser?: { headless: boolean; slowMoMs: number };
}

function log(message: string): void {
  process.stderr.write(`[agent] ${message}\n`);
}

function describe(error: unknown): string {
  return error instanceof Error ? (error.stack ?? error.message) : String(error);
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString('utf8');
}

function parseRequest(raw: string): AgentRequest {
  const request = JSON.parse(raw) as AgentRequest;
  if (
    typeof request.runId !== 'string' ||
    typeof request.task?.instruction !== 'string' ||
    typeof request.browser?.headless !== 'boolean' ||
    request.budgets === undefined ||
    request.artifacts === undefined
  ) {
    throw new Error('request missing required fields');
  }
  return request;
}

function zeroMetrics(): Metrics {
  return {
    modelCalls: 0, inputTokens: 0, outputTokens: 0, wirCalls: 0, browserDeliveries: 0,
    providerLatencyMs: 0, observationBytes: 0, cachedInputTokens: null, estimatedCost: null, reasoningTokens: null,
    // A crash before the loop ran: no reads, and the arm is still whatever this
    // process was configured for — a crashed episode must not read as the baseline.
    readAfterRead: 0, readScreenshot: process.env['WIR_READ_SCREENSHOT'] === '1',
    toolCallsIssued: 0, batchedTurns: 0, unmovedTurns: 0,
    buildMtime: null, headSha: null,
    temperature, seed, modelId: modelName,
  };
}

let wrote = false;

async function writeResponse(response: AgentResponse): Promise<void> {
  if (wrote) return;
  wrote = true;
  await new Promise<void>((resolve) => {
    process.stdout.write(JSON.stringify(response), () => resolve());
  });
}

async function main(): Promise<void> {
  let response: AgentResponse;
  try {
    const request = parseRequest(await readStdin());
    parsedRunId = request.runId;
    parsedHeadless = request.browser.headless;
    const runId = parsedRunId;
    log(`run ${runId}: task ${request.task.taskId} (${request.task.expectedAction})`);
    const result = await runEpisode(request);
    log(`run ${runId}: status ${result.status}, ${result.metrics.modelCalls} model calls, ${result.metrics.wirCalls} wir calls`);
    response = {
      protocolVersion: PROTOCOL_VERSION,
      runId,
      status: result.status,
      finalResponse: result.finalResponse,
      metrics: result.metrics,
      effectiveBrowser: result.effectiveBrowser,
    };
  } catch (error) {
    log(`fatal: ${describe(error)}`);
    response = {
      protocolVersion: PROTOCOL_VERSION,
      runId: parsedRunId,
      status: parsedRunId === '' ? 'invalid_request' : 'agent_failed',
      finalResponse: {},
      metrics: zeroMetrics(),
      // Echo what we know (review: the error path omitted effectiveBrowser and
      // the runner's conflict check ran first, misclassifying agent failures
      // as agent_protocol_invalid).
      ...(parsedHeadless !== null
        ? { effectiveBrowser: { headless: parsedHeadless, slowMoMs: 0 } } : {}),
    };
  }
  await writeResponse(response);
  process.exit(0);
}

// What the request asked for, once parsed — so even crash responses can echo it.
// The runId is here for the same reason parsedHeadless is: the harness discards
// any response whose runId does not echo the request (agent_process.py's
// _parse_response), so a crash response that hardcoded '' was classified
// agent_protocol_invalid instead of agent_process_failed — the crash safety net
// discarding itself. A crash BEFORE the parse leaves it empty, and
// protocol-invalid is then correct: no valid request ever existed.
let parsedRunId = '';
let parsedHeadless: boolean | null = null;

function crashResponse(): AgentResponse {
  return {
    protocolVersion: PROTOCOL_VERSION,
    runId: parsedRunId,
    status: 'agent_failed',
    finalResponse: {},
    metrics: zeroMetrics(),
    ...(parsedHeadless !== null
      ? { effectiveBrowser: { headless: parsedHeadless, slowMoMs: 0 } } : {}),
  };
}

// The one-JSON-response contract must survive crash classes the promise chain
// never sees — ENOSPC/EPIPE on the trajectory stream killed the process with
// zero bytes on stdout (review, cluster D). The `wrote` flag makes this safe:
// if the real response already went out, these are no-ops.
process.on('uncaughtException', (error) => {
  log(`uncaught: ${describe(error)}`);
  void writeResponse(crashResponse()).catch(() => undefined).then(() => process.exit(0));
});
process.on('unhandledRejection', (reason) => {
  log(`unhandled rejection: ${describe(reason)}`);
  void writeResponse(crashResponse()).catch(() => undefined).then(() => process.exit(0));
});

main().catch(async (error: unknown) => {
  log(`unhandled: ${describe(error)}`);
  await writeResponse(crashResponse()).catch(() => undefined);
  process.exit(0);
});
