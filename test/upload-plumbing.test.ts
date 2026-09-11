// The producer half of upload. Core's fence (core/act.ts resolveUpload) shipped
// complete in 2747b32, but NOTHING produced the directory: AgentRequest had no
// uploadDir field and startSession forwarded none, so every model `upload` was
// invalid_args regardless of the page or the file — and the fence's own refusal
// reads like the model's mistake, so the episode burns calls retrying it.
// A session-level test cannot pin this defect: WirSession.start accepted
// uploadDir all along; only the spawned agent binary exercises the producer path.
//
// Proven live: BU-bench V1 task c1f60ee6 (external adapter, agent gave up
// 2026-08-11 after the date field and the upload both failed). Artifact
// (external to this repo and prunable, so the load-bearing line is quoted):
//   /Users/iavinas/only/benchmark/bu-bench/run_data/BU_Bench_V1_framework_wir_
//   browser_local_headful_model_mimo-v2.5_start_at_20260811_134717/wir_episodes/task_2
// Verbatim rejection on every upload call: "no upload directory was declared
// for this episode, so there are no files to attach".
import { strict as assert } from 'node:assert';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { runAgent, toolCall } from './trajectory-rejections.test.js';

const PAGE = '<!doctype html><title>u</title><h1>Upload</h1>' +
  '<form><label>Attachment <input type="file" name="attachment"></label></form>';

/** A function turn reads the runtime-minted ref out of the stub's own
 *  transcript, exactly as the model does — refs are opaque, and a hardcoded one
 *  would assert on an identity the runtime is free to change. A file input
 *  projects as role "button" carrying its label's accname ("Attachment"), so the
 *  ref is matched by that adjacent ref/role/name triplet, tolerating the
 *  transcript's JSON escaping; the fallback is the body's last ref. */
function uploadFromTranscript(body: string): unknown {
  const input = body.match(
    /(n_[0-9a-f]{12,40})\\?",\\?"role\\?":\\?"button\\?",\\?"name\\?":\\?"Attachment/);
  const all = [...body.matchAll(/n_[0-9a-f]{12,40}/g)].map(m => m[0]);
  const ref = input?.[1] ?? all[all.length - 1];
  return toolCall('act', JSON.stringify({ ref, action: 'upload', value: 'report.txt' }));
}

const uploadRecord = (trajectory: Record<string, unknown>[]): Record<string, unknown> | undefined =>
  trajectory.find(r => r['kind'] === 'wir'
    && (r['request'] as Record<string, unknown> | undefined)?.['action'] === 'upload');

test('browser.uploadDir travels the wire: a declared directory makes upload attach', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'wir-upload-'));
  writeFileSync(join(dir, 'report.txt'), 'proof\n');
  const { trajectory } = await runAgent({
    html: PAGE, uploadDir: dir, maxModelCalls: 3,
    turns: [toolCall('read', '{}'), uploadFromTranscript,
            toolCall('give_up', '{"reason":"test over"}')],
  });
  const upload = uploadRecord(trajectory);
  assert.ok(upload, `the upload act must reach dispatch: ${JSON.stringify(trajectory)}`);
  const response = upload['response'] as Record<string, unknown>;
  const effect = response['effect'] as Record<string, unknown> | undefined;
  assert.equal(effect?.['evidence'], 'file_attached',
    `attaching is the local effect the fence was guarding: ${JSON.stringify(response)}`);
  assert.equal(effect?.['verdict'], 'verified',
    'the page\'s own readback of the input, not our belief about what we sent');
});

test('without uploadDir the fence still refuses — default deny survives the producer', async () => {
  const { trajectory } = await runAgent({
    html: PAGE, maxModelCalls: 3,
    turns: [toolCall('read', '{}'), uploadFromTranscript,
            toolCall('give_up', '{"reason":"test over"}')],
  });
  const upload = uploadRecord(trajectory);
  assert.ok(upload, `the refused upload must still be recorded: ${JSON.stringify(trajectory)}`);
  assert.match(JSON.stringify(upload['response']),
    /no upload directory was declared/,
    'the fence\'s own sentence — first regression coverage it has ever had');
});
