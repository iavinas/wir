// The act action enum has outrun its prose twice: the CLAUDE.md action list said
// three until 2026-08-11 and seven until 2026-08-12, and hover (8cce549) landed
// in the enum while BASE_SYSTEM_PROMPT's action list stayed at seven. The enum
// (core/toolschemas.ts) is the contract; the tool description and the system
// prompt are the only places a model learns it, so an action missing from either
// is invisible to the model. This pin turns the next omission into a suite
// failure. Offline: no browser, no provider, no network.
import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { toolDefinitions } from '@wir/core';
import { BASE_SYSTEM_PROMPT } from '../agent/loop.js';

test('every act enum member appears in the act description and the prompt action line', () => {
  const act = toolDefinitions.find((tool) => tool.function.name === 'act');
  assert.ok(act, 'the act tool is defined');
  const properties = act.function.parameters['properties'] as
    Record<string, { enum?: string[] } | undefined>;
  const actions = properties['action']?.enum ?? [];
  assert.ok(actions.includes('click'), 'the enum was found where the schema keeps it');

  const promptLine = BASE_SYSTEM_PROMPT.split('\n').find((line) => line.startsWith('act — '));
  assert.ok(promptLine, 'the prompt lists act with its actions');

  for (const action of actions) {
    // "name — " pins a purpose clause, not an incidental mention of the word.
    assert.ok(act.function.description.includes(`${action} — `),
      `act's description explains ${action}`);
    assert.ok(promptLine.includes(action),
      `the prompt's act line mentions ${action}`);
  }
});
