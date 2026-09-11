// The read-screenshot experiment (docs/plans/read-screenshot-experiment.md) is
// measured by diffing two arms of the same suite, so two things must be true or
// every number it produces is noise:
//
//   1. flag OFF is byte-identical to the code without the experiment — otherwise
//      the "baseline" arm is not the baseline, and the comparison measures the
//      instrument rather than the projection;
//   2. a truncated image FAILS LOUDLY at the send site — the reported prior
//      failure was a clipped base64 payload reaching the model, which produces a
//      confused episode and a wrong number with nothing in the logs to explain it.
//
// Offline: no browser, no provider, no network.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { toolDefinitions, toolDefinitionsFor } from '@wir/core';
import { stripImageParts } from '../agent/telemetry/redact.js';

test('flag off leaves the tool definitions byte-identical', () => {
  assert.equal(
    JSON.stringify(toolDefinitionsFor({ readScreenshot: false })),
    JSON.stringify(toolDefinitions),
  );
});

test('flag on changes the read description and nothing else', () => {
  const on = toolDefinitionsFor({ readScreenshot: true });
  assert.equal(on.length, toolDefinitions.length);
  for (const [index, tool] of on.entries()) {
    const base = toolDefinitions[index] as (typeof toolDefinitions)[number];
    assert.equal(tool.function.name, base.function.name);
    assert.deepEqual(tool.function.parameters, base.function.parameters);
    if (tool.function.name === 'read') {
      assert.ok(tool.function.description.startsWith(base.function.description),
        'the clause is appended, never a rewrite of what read already promised');
      assert.ok(tool.function.description.length > base.function.description.length);
    } else {
      assert.equal(tool.function.description, base.function.description);
    }
  }
});

// A JPEG ends with FFD9. Cut the payload anywhere and the terminator goes with it,
// which is what makes the check need no expected length and no side channel.
function jpegDataUri(bytes: number[]): string {
  return `data:image/jpeg;base64,${Buffer.from(bytes).toString('base64')}`;
}

const INTACT = [0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0xff, 0xd9];

test('an intact image passes the send-site check and a truncated one does not', async () => {
  const { chatComplete } = await import('../agent/provider.js');
  const message = (url: string) => [{
    role: 'user' as const,
    content: [{ type: 'image_url' as const, image_url: { url } }],
  }];

  // Truncated: the terminator is gone. Must throw BEFORE any network call, so a
  // rejection here proves the check fired rather than the provider being absent.
  await assert.rejects(
    () => chatComplete(message(jpegDataUri(INTACT.slice(0, -2))), []),
    /truncated/,
  );

  // A payload clipped mid-quantum fails on the length invariant alone.
  await assert.rejects(
    () => chatComplete(message(`${jpegDataUri(INTACT)}X`), []),
    /truncated/,
  );

  // Intact: the image check passes, so failure now comes from the network (there
  // is no provider in a unit test). Anything mentioning truncation would be a
  // false positive on a good image — the defect that makes the check unusable.
  await assert.rejects(
    () => chatComplete(message(jpegDataUri(INTACT)), []),
    (error: Error) => !/truncated/.test(error.message),
  );
});

test('telemetry keeps the image shape but never the bytes', () => {
  const url = jpegDataUri(INTACT);
  const stripped = stripImageParts([
    { type: 'text', text: 'Page after read 3 (epoch ABC).' },
    { type: 'image_url', image_url: { url, detail: 'high' } },
  ]) as Record<string, unknown>[];

  assert.equal(stripped[0]?.['text'], 'Page after read 3 (epoch ABC).',
    'the label survives — it is what makes the span readable');
  const image = stripped[1]?.['image_url'] as Record<string, unknown>;
  assert.equal(image['detail'], 'high');
  assert.match(String(image['url']), /^\[image:jpeg b64len=\d+\]$/);
  assert.ok(!JSON.stringify(stripped).includes(url.split(',')[1] as string),
    'no fragment of the payload reaches the vendor');
});

// The seam that actually ships to the vendor. Asserting stripImageParts alone
// would prove the helper works while the telemetry path bypassed it — and the
// credential opt-out must NOT re-enable images: it is about secrets, not size.
test('the telemetry seam strips images even with the value opt-out set', async () => {
  const { redactGenerationRecord } = await import('../agent/telemetry/redact.js');
  const url = jpegDataUri(INTACT);
  const payload = url.split(',')[1] as string;
  const record = {
    model: 'test', ms: 1, usage: { input: 1, output: 1 },
    input: { messages: [
      { role: 'user', content: [{ type: 'image_url', image_url: { url } }] },
    ] },
    output: { role: 'assistant', content: null },
  };

  const previous = process.env['WIR_TELEMETRY_LOG_VALUES'];
  try {
    for (const optOut of [undefined, '1']) {
      if (optOut === undefined) delete process.env['WIR_TELEMETRY_LOG_VALUES'];
      else process.env['WIR_TELEMETRY_LOG_VALUES'] = optOut;
      const shipped = JSON.stringify(redactGenerationRecord(record));
      assert.ok(!shipped.includes(payload),
        `payload reached the vendor with WIR_TELEMETRY_LOG_VALUES=${String(optOut)}`);
      assert.match(shipped, /\[image:jpeg b64len=\d+\]/);
    }
  } finally {
    if (previous === undefined) delete process.env['WIR_TELEMETRY_LOG_VALUES'];
    else process.env['WIR_TELEMETRY_LOG_VALUES'] = previous;
  }
});

test('stripping is a no-op on the string content every other message uses', () => {
  assert.equal(stripImageParts('{"documentEpoch":"A"}'), '{"documentEpoch":"A"}');
  assert.equal(stripImageParts(null), null);
});
