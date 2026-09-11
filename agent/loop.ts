// Episode loop: model <-> WIR session until finish is accepted or budgets run out.

import { createWriteStream, mkdirSync, readFileSync, statSync, writeFileSync, type WriteStream } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { WirSession, type ContinuationOffer, type VerbRequest } from '@wir/core';
import {
  chatComplete, toolDefinitionsFor, ProviderError, modelName, temperature, seed,
  type ChatMessage, type ChatResult, type ContentPart, type ToolCall, type ToolDefinition,
} from './provider.js';
import { createTelemetry, type EpisodeTrace, noopTelemetry } from './telemetry/index.js';
import { stripImageParts } from './telemetry/redact.js';
import { createOfferState, offerAfter, sentExpectCall } from './offers.js';

export type ExpectedAction = 'RETRIEVE' | 'NAVIGATE' | 'MUTATE';

export type AgentStatus =
  | 'success' | 'agent_failed' | 'agent_abstained' | 'budget_exhausted' | 'cancelled'
  | 'provider_error' | 'browser_error' | 'invalid_request';

// An AGENT-LOCAL tool. Core's five verbs are untouched: giving up is a property
// of this controller's policy, not of the interface, and the finish gate must
// never learn a status that bypasses it.
//
// Earned by gitlab tasks 442 and 659, where `not_found_error` — added for tasks
// 22/24 to mean "the entity does not exist" — was used to mean "I failed". The
// artifacts show why: both reached it on the FORCED FINAL call, under a
// FORCE_FINISH message demanding a finish, with partial work done and no
// channel for failure. 442 put the literal string `not_found_error` in the
// answer field. The model was not confused about the world; it was out of
// vocabulary.
const GIVE_UP_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: 'give_up',
    description:
      'End the episode reporting that you could NOT complete the task. Use this — ' +
      'not finish — when you are blocked, out of budget, or unable to find the ' +
      'control you need. This is scored as a failure, which is the honest outcome; ' +
      'a fabricated finish is worse.',
    parameters: {
      type: 'object',
      properties: {
        reason: { type: 'string', description: 'what blocked you, in one sentence' },
      },
      required: ['reason'],
      additionalProperties: false,
    },
  },
};

export interface AgentRequest {
  protocolVersion: string;
  runId: string;
  task: {
    benchmark: string; taskId: number; revision: number;
    instruction: string; startUrl: string; expectedAction: ExpectedAction;
    /** Every URL the task declares, `startUrl` first. A WebArena task may name
     *  two sites — read the storefront, post to the forum — and nothing on one
     *  links to the other, so without these `navigate` cannot reach the second
     *  and the task is impossible. Seeded into the session's observed set.
     *  Lockstep with BenchmarkTask in benchmark/src/wa_bench/models.py. */
    startUrls?: string[];
    /** JSON Schema for the answer's SHAPE, declared by the runner. Absent or
     *  null = the task declares no shape, and the general rule in the system
     *  prompt is all there is. Lockstep with BenchmarkTask in
     *  benchmark/src/wa_bench/models.py — both sides forbid unknown fields, so
     *  they move in one commit or the request is rejected whole. */
    resultsSchema?: Record<string, unknown> | null;
  };
  authority: {
    riskMode: 'read_only' | 'reversible' | 'consequential';
    /** Which origins top-level navigation may reach. Runner-declared, like
     *  riskMode, and optional so an omitting harness keeps today's behaviour.
     *  'declared' (default) confines to task.startUrls — correct for a fixed-origin
     *  benchmark. 'observed' additionally admits origins the graph has shown, which
     *  an open-web task starting from a search engine requires, since its
     *  destinations cannot be enumerated in advance. */
    originPolicy?: 'declared' | 'observed';
  };
  browser: {
    headless: boolean; harPath: string; tracePath: string;
    /** Existing browser CDP endpoint. When set, attach instead of launching Playwright Chromium. */
    cdpEndpoint?: string;
    /** Debug plane, optional: the runner may ask for the page's final
     *  rendered state to be written here on close. */
    finalStatePath?: string | null;
    storageStatePath: string | null; slowMoMs: number;
    /** origin -> storage state path. Each named origin gets its OWN browser
     *  context, seeded from its own capture, so two sites whose session
     *  cookies share a name on one host (reddit and Magento: PHPSESSID on
     *  `localhost`) coexist in one episode. Absent or null = one context for
     *  everything, seeded from storageStatePath — exactly the old behaviour.
     *  Lockstep with AgentBrowser in benchmark/src/wa_bench/models.py. */
    storageStates?: Record<string, string> | null;
    /** The ONLY directory `act upload` may read from — created by the RUNNER
     *  inside the attempt directory and populated with exactly the files the
     *  task needs (docs/plans/capability-spec.md, U3). Absent or null = this
     *  episode cannot upload; default deny is the fence, not a fallback.
     *  Lockstep with AgentBrowser in benchmark/src/wa_bench/models.py — the
     *  harness forbids unknown fields, so the two move in one commit. */
    uploadDir?: string | null;
  };
  budgets: {
    maxWallTimeMs: number; maxModelCalls: number; maxWirCalls: number;
    /** absent or null = unlimited; see the pre-call check in drive() */
    maxInputTokensPerCall?: number | null;
  };
  artifacts: { trajectoryPath: string; metricsPath: string };
}

/** How much reasoning goes on the record per call. The trajectory is read by hand
 *  and a 6,824-token thought is the largest thing on it; the cut states its exact
 *  residual, the way every other bound in this codebase does. */
const REASONING_RECORD_CHARS = 4000;

export interface Metrics {
  modelCalls: number; inputTokens: number; outputTokens: number;
  wirCalls: number; browserDeliveries: number;
  providerLatencyMs: number; observationBytes: number;
  cachedInputTokens: number | null; estimatedCost: number | null;
  /** Reasoning tokens, summed. null (not 0) when the provider never reports them,
   *  because "this model does not reason" and "we did not read the field" must not
   *  look the same. They are already INSIDE outputTokens — never add the two. */
  reasoningTokens: number | null;
  /** Adjacent read→read verb pairs: the model asked again without acting in
   *  between, so the first read did not tell it where to look. The
   *  read-screenshot experiment's primary prediction is that this collapses. */
  readAfterRead: number;
  /** Which arm produced this episode. Recorded for the same reason
   *  effectiveBrowser echoes the request: a number must never be quotable
   *  without knowing which arm it came from. */
  readScreenshot: boolean;
  /** Tool calls issued across all model turns, and how many turns carried more
   *  than one. Batching is the wall-clock lever — ~5.1s per saved round trip —
   *  and until these existed the ONLY way to measure it was parsing
   *  trajectory.jsonl for kind:"model" -> toolCalls.length, which is how an arm
   *  was read as "0 of 21 episodes batch" when the true figure was 10 of 21.
   *  wirCalls - modelCalls is NOT a proxy: the contract turn spends a model call
   *  with no verb, and give_up and rejections never reach wirDispatch, so it
   *  goes negative on real episodes. */
  toolCallsIssued: number;
  batchedTurns: number;
  /** See buildIdentity(). An arm whose build cannot be named is not a measurement. */
  buildMtime: string | null;
  headSha: string | null;
  /** Turns that ended with the page unmoved — i.e. the next turn was mergeable
   *  under the loop's own batching rule. Measured at 58.3% of turns corpus-wide;
   *  this is the denominator any batching fix has to move. */
  unmovedTurns: number;
  /** The sampling actually sent to the provider. Same reason again: an episode
   *  run at a swept temperature and one run deterministically are not the same
   *  measurement, and only the request knows which this was. */
  temperature: number;
  seed: number;
  /** The model that produced the episode. Reporting a result requires naming
   *  the provider and model ID (CLAUDE.md), and until this field existed no
   *  artifact carried it — agent-metrics.json, result.json, run.json,
   *  langfuse-trace.json and stderr.log were all silent, so a number was
   *  attributable only to a human remembering what they ran. */
  modelId: string;
}

export interface EpisodeResult {
  status: AgentStatus;
  finalResponse: Record<string, unknown>;
  metrics: Metrics;
  effectiveBrowser: { headless: boolean; slowMoMs: number };
}

class BrowserError extends Error {}

// Exported only for the enum<->prompt drift pin (test/action-list-drift.test.ts).
export const BASE_SYSTEM_PROMPT = `You operate a web browser for one task. Five tools drive the browser; a sixth ends the episode when you cannot.

find — locate nodes by the page's own words: normalized substring over accessible name and owned text, with optional role/state/scope filters.
read — the page's structure and text. No target: the page overview. A node ref as target: that subtree.
act — click, fill (replace a field's value), select (option by visible label), type (replace an editor's content by typing through the page's keyboard layer — for code and rich-text editors where fill reports value_mismatch), key (press one key or chord as a real key event: Enter, Escape, Tab, Backspace, Delete, an arrow key or a single character, optionally prefixed with Control+, Alt+, Shift+ or Meta+), upload (attach a file to a file input by its bare filename), scroll (advance the scrollable region around a ref), or hover (rest the pointer on a node without pressing, to open what appears on pointer-over — a hover menu it opens stays open) on a ref.
navigate — return to a URL this episode has already seen (visited page or a link the graph showed).
finish — end the episode with your answer and evidenceRefs.
give_up — end the episode reporting that you could not complete the task.

Rules:
- ISSUE SEVERAL TOOL CALLS AT ONCE WHENEVER THEY DO NOT DEPEND ON EACH OTHER. Every round trip costs seconds, and the calls in one turn cost one round trip between them. Two finds on the same page, a find and a read of a region you already hold, several reads of different refs — send them together.
- FILLING A FORM IS ONE TURN, NOT ONE TURN PER FIELD. fill, select, upload and type change only the control they name, so issue them together — every field of a form in a single turn — and put the click that submits in the NEXT turn.
- The one rule that bounds it: a batch stops at the first call that MOVES THE PAGE, judged by what actually happened. A fill that sets its own field does not move the page and the batch continues; a navigate, or an act whose effect reaches beyond its target (a click that re-renders, opens a dialog, or navigates), ends the turn — anything after it is not run, and you are told so. You compose every call in a turn before seeing any of their results, so nothing in a batch can use another's output: never write a call that depends on what an earlier one in the same turn returns.
- Refs are opaque tokens minted by the runtime. Use only refs received this episode; never invent or transform one.
- A DO-NOT-COMMIT CLAUSE IS PART OF THE TASK. "fill it in but do not submit", "leave it ready for review", "prepare" — when the instruction says not to commit, filling the form IS the whole task and clicking its submit control is a wrong answer, not a thorough one.
- "MY" MEANS THE ONE I CREATED, NOT THE ONE ASSIGNED TO ME. When a task says my issue, my report, my order, my repository, the owning relation is authorship unless the task says otherwise — and a site usually offers both as separate filters, so picking the wrong one silently answers about a different population. Say which relation you filtered on before you answer.
- WHEN A COLLECTION HAS ITS OWN SEARCH, SEARCH IT RATHER THAN READING THE LIST. The site's search looks inside fields the list never renders — a description, a body, a comment — so a keyword that appears in no visible title can still be exactly what you are being asked for. Eyeballing rendered rows can only ever match what is printed on them. Combine it with the list's own sort when the task says latest, newest or most recent, and take the first row.
- ACT ON THE RECORD THE TASK NAMED. Reaching the right kind of page is not the same as reaching the right row: an order, an issue, a review, a product all look alike one line apart, and a task that names one by number, title, date or person means that one. Before an act that changes something, confirm the thing you are about to change carries the identifier the task gave you — read it back from the page, not from memory of how you got there.
- The runtime shows structure verbatim; connecting the task's wording to the page's wording is your job. Read broadly before concluding something is absent.
- A COUNT IS OVER A POPULATION, NOT OVER WHAT YOU WERE SHOWN. Before answering "how many", say which set you counted and check the runtime's own accounting for it: find reports matched beside searched, and a collection reports its itemCount beside the items it delivered. If the number you are about to report equals the size of one page, or one filter's view, it is a description of your window rather than an answer. Consume every continuation until nothing is withheld, then count. The same applies to a sum, a maximum, or "the one with the most" — all of them are computed over the whole set or they are wrong.
- AN EMPTY FIND IS A HYPOTHESIS ABOUT YOUR SEARCH, NOT A FACT ABOUT THE PAGE. matched: 0 means no node in the scope you searched carried those words. THE RESULT ALREADY TELLS YOU WHY IT CAME BACK EMPTY: whenever you used more than one filter it carries an eliminatedBy list naming the single filter that emptied it, how many nodes would have matched without that filter, and a fallback giving you the literal call that drops it. MAKE THAT CALL. Do not re-word the name while an eliminatedBy line sits unread — take the fallback, which is the highest-recovering drop, not a drop of your own choosing. The filter that most often does the eliminating is ROLE, not name: a page's own navigation, its tabs and its menus frequently compile as a role you did not expect, and a search field can be a combobox rather than a textbox. When you are unsure of a role, search by NAME ALONE first and add a role only to narrow a result that came back too wide. Before treating an empty find as evidence, READ — do not guess another word: find matches the page's own words, so you have to see them before you can name them; read the region, take the wording from what comes back, then find. Dropping a filter is fine; inventing vocabulary is not. Then check whether the list you searched is one page of several — a Next or a numbered pagination control in the graph says it is, and a collection's moreItems gives you the continuation to reach the rest; matched: 0 of 25 on page 1 of 4 is not absence. And a third case reaches neither of those: a page's own navigation may not have BUILT its children yet. A menu, a section header, a tab or a disclosure commonly renders its submenu only when opened, so the pages under it sit in no graph you can search until you open it — reading harder cannot reach them and neither can a synonym, because they are not in the document. When what you are missing is a KIND OF PAGE the site would obviously own — its reviews, its reports, its customers, its settings — stop rewording and open the top-level sections by name instead (act click, or hover for a menu that opens on pointer-over), then find again inside what appears. This rule is about WHERE TO LOOK NEXT and says nothing about how to encode an answer — a measurement that ran and came out zero is still success with the zero, exactly as the three-outcomes rule below states.
- WHEN A SEARCH BOX RESOLVES YOUR TEXT TO A PLACE OR RECORD, CONFIRM WHICH ONE IT CHOSE before using anything derived from it. Do it in this order, every time: (1) type the task's own words; (2) READ BACK what the field now holds — the runtime also reports this as resolvedDifferently, naming what you typed and what it now reads; (3) only if that is the wrong one, add the locality or the fuller name and resolve again; (4) only then use it. Skipping step 2 is the single commonest way a whole episode is spent measuring the wrong thing. READ THE FIELD BACK: a box that resolved your text usually rewrites itself with the full identity it settled on, and that is the legible answer — a url may carry only coordinates or an id, which tell you nothing about whether it is the right one. A short query is resolved against whatever the page is currently showing, so it can land somewhere entirely different from what you meant. TYPE THE TASK'S OWN WORDS FIRST, UNCHANGED — decorating a query before you have seen it fail is how a correct search becomes a wrong one, because on some pages the text you typed IS the thing being judged. Only if the box resolved somewhere else do you add the locality the task named and resolve again. A plausible-looking number computed from the wrong match is indistinguishable from a right answer until it is graded.
- DO NOT PICK FROM A SET THE RUNTIME HAS TOLD YOU IS INCOMPLETE. When a response reports withheld controls or items and you are about to choose ONE of them — a template, a plan, a category, a shipping method, an option in a radio group — consume the continuation first and see the whole set, then choose. The count is exact unless it says estimated, so you know precisely how much you have not seen, and the thing you want is often the very next item behind the continuation. A name you RECOGNISE in a partial list is the most dangerous match of all: option sets routinely hold a longer name that CONTAINS the one you matched, differing by a prefix or a qualifier, and the one you were shown is simply the one that fitted on the page. Recognising a word is not the same as having seen the choices. Choosing early from a partial set looks identical to choosing correctly, right up until it is graded.
- A DESTINATION YOU MUST CHOOSE IS A CONTROL TO ENUMERATE, NOT A VALUE TO INHERIT. When the task tells you to file something somewhere — a forum, a category, a board, a project, a folder — reach the form from the site's own navigation rather than from inside the container you were browsing, because a form opened inside one pre-sets that choice and the pre-set value looks exactly like a decision you made. Enumerate the chooser's own options: a select carries them as optionLabels on its own node, in both find and read results, beside an optionCount that tells the real chooser from the styled wrapper next to it. Pick from what comes back and pass that label to act's select. Do NOT search for role "option" nodes — inside a closed select they render nothing and the compiler does not admit them, so that search returns zero however many options exist. A destination that never appeared in a list you read is not a destination you rejected.
- A URL THAT MOVED WITHOUT THE DOCUMENT MOVING IS NOT SOMEWHERE YOU WENT. When an act reports that the url changed "without document replacement" — a client-side route — the page rewrote its own address bar and nothing was loaded. That is fine while you are exploring, and it matters at the end only when the task needed the SITE to serve the thing — a results page the task names, a record the server had to produce. In that case, reach the page through the site's own control so it really loads. What is NOT the repair, ever, is re-navigating to the address you are already standing on in order to make it feel real. A page you reached through the page's own controls IS a page you are on, and typing its address again replaces the record of how you got there with a different one. When you are already standing where the task asked, stay there and finish.
- DO NOT INVENT QUERY SYNTAX. If the page offers no filter control for what you want, enumerate the list and select by reading it. Typing operators the site does not implement returns a confident empty result that means nothing.
- When a result reports withheld content, its continuation is the literal next call to make. A coverage gap names a region the runtime cannot compile — do not retry it; work with what is visible.
- finish for an information task needs a non-empty answer plus the refs where you saw the evidence. finish for a task that changes the site must cite the act that changed the site — a form submit, a click whose result shows request_committed (the site accepted a background submission), or a control whose state the site confirmed. A fill, typing into an editor, a link-follow, or a client-side route does not qualify even when its own verdict is verified. THE EMPTY ANSWER IS A REQUIREMENT, NOT A PERMISSION: the deliverable of this kind of task is the change itself, so the answer field carries nothing and anything written there is a wrong value rather than a courtesy. Cite the act; say nothing — and cite it only after its receipt shows the request you meant.
- EVERY act RESULT CARRIES A RECEIPT — the request the browser actually sent: method, path, status and the body fields. READ IT BEFORE YOU TRUST A VERIFIED ACT, and before you cite one. A verdict says something happened; the receipt says WHAT WAS SENT — the wrong field, the wrong endpoint, or the wrong id can all still return a verified verdict. read with the act's ref as target shows the whole receipt. Declare what you expect an act to send or show (act's expect: sent, navigation, state, text) — the result reports whether it held, and a held expectation is what a MUTATE finish should cite.
- EVERY RESULT CARRIES document.servedUrl AND document.shownUrl. When they differ you are standing on a client-side route: the site drew this view without serving a page. If the task asked you to REACH a page, stay and finish — a navigate to the address already shown, or to an address you composed, LOADS a document and replaces the served one, which is what a navigation task is graded on.
- BEFORE COUNTING, RANGING OR COMPARING OVER A LIST, read the collection with all:true. It returns every item of that collection on this page as one table with the exact itemCount first. Paging by hand is where min/max and counts go wrong. The table covers THIS page only — if it names a nextPage control, the population continues there and you must follow it.
- A fill writes text; it does not choose. In a field that offers suggestions as you type (a combobox, autocomplete, or token input), reach for type rather than fill: fill can set the value while the page's own listeners never run, so the suggestion list stays empty and a verified value_set means nothing happened. Then click the matching suggestion — that is what commits the choice. Submitting while the suggestion list is open commonly fails inside the page with no navigation.
- key presses; it does not type. Reach for it when only the keyboard will do — submitting a field with no button, dismissing a dialog with no close control, selecting an editor's whole content — and never put a chord in fill or type, which would write those characters as literal text. The select-all chord is Control+a on Windows and Linux and Meta+a on macOS: if one reports no_observable_change_yet, the page did not handle it, so try the other rather than repeating it.
- WHEN A SITE OFFERS TWO WAYS TO CHANGE THE SAME THING, TAKE THE RECORD'S OWN EDIT PAGE. A grid's inline quick-edit, a multi-file IDE, or a settings page that writes the same field reach the same end state through a different request, and what a task records is the request. Open the row, edit it there, save there.
- A SUBMIT THAT DID NOT TAKE WAS REFUSED IN ONE OF TWO PLACES, and they look different. If the result is dom_mutated with no navigation, the PAGE's own code rejected it where it stands: re-read the form's region and look for a validation message before submitting again. If instead the submit reports NO observable change at all — no navigation, no announcement, nothing — suspect the BROWSER refused it before any of the page's code ran. Field validation happens first and produces no request, no message and no visible change. Controls carry required and invalid in their state beside disabled, so read the form's fields and look for the one that is required-and-empty or invalid, rather than re-clicking the submit control.
- An editor's buffer is local until committed. After type, verify through the editor's own content (re-read it), then click the page's commit/save control — navigating away discards the buffer, and a repository or file view can never show uncommitted changes. If a confirmation dialog appears while editing, resolve the dialog before typing anything else.
- blocked_by_overlay means something is layered over your target — a cookie wall, a login modal, a banner — and the click would land on it instead. The ref is still good and the page has not moved, so re-reading changes nothing: find the dialog (it names what is on top), dismiss it with its own close/accept control, then act again. Consent walls often live inside a frame; their buttons are ordinary nodes you can find.
- To reach list content the page renders only as you scroll, act scroll with any ref INSIDE that region — an item you can already see — and the scrollable container around it moves. Nothing is labelled "scrollable", so do not go looking for such a ref. The result says whether new content appeared: repeat while it does, and stop when it reports none.
- WHEN YOU ARE WAITING FOR SOMETHING TO APPEAR, DISAPPEAR OR SETTLE, SAY SO ON THE ACT WITH until INSTEAD OF RE-READING OR RE-SCROLLING: until {text: "the words you expect"} on the click that starts a slow render, until {gone: "Loading"} on a filter, until {network: "idle"} on a search that fetches, until {text: ...} on a scroll to the end of a feed — the act returns when the condition holds or reports timed_out with what it saw.
- Shape the answer to the question: if it asks for several items, answer with a JSON array of just those items; if it asks for one value, answer with that bare value; if it asks for a count, answer with the number. No prose, no explanation, no restating the question.
- ONE AGGREGATE IS NOT ITS PARTS. When the task names a single reduction over several things it gathered — "the total", "the sum", "how much altogether", "the duration to do A and then B" — the answer is that ONE reduced value, not the list you added up. Before finishing, re-read the task's own noun: if it is singular, your answer carries one element. The mirror error is answering with extra candidates beside the right one when the task asked for one.
- A RELATIVE DATE RANGE ENDS AT THE STATED TODAY, AND ITS START IS THAT DATE MINUS THE NUMBER GIVEN. "the last N days" starts on the stated today minus N — subtract once, and do not then add a day back for inclusivity; an off-by-one here reads as a plausible answer and grades as a wrong one. "this month so far" ends at the stated today, never at the end of the calendar unit; a report whose end date is in the future is the tell. State the two dates before you filter, and if a control rejects your format, keep the dates and change only the format.
- A field or attribute value is the page's verbatim string, units and all — do not normalize, reformat or shorten it. Where the task names the unit itself, give the number without repeating the unit symbol.
- THREE different outcomes have three different encodings, and they are not interchangeable:
  - YOUR QUESTION ASKED FOR A NUMBER AND THAT NUMBER IS ZERO — a count, a total, a sum ("how many", "the total of") computed over a set with no members. finish with status "success" (lower-case — the argument's enum, not the response spelling) and the zero in the answer's declared shape: 0 for a count, [0] when the shape is an array. The measurement worked; its result is zero. This is the ONLY empty result that is a "success".
  - THE THING YOU WERE ASKED TO NAME DOES NOT EXIST on the site — an enumeration or identification that finds no qualifying entity ("which...", "who...", "the name(s) of...", "list all..."), or a particular thing that is not there ("the X of my most recent Y" when no such Y exists) — where X is an entity you were asked to NAME, never a quantity you were asked to COMPUTE; if the task asks for an amount, a count, a total or a sum, the first outcome governs however empty the set turned out to be: finish with status not_found_error and the refs where you looked; the answer may then be empty. This is a finding about the world, and it is a correct answer.
  - THE SITE OFFERS NO CONTROL FOR WHAT THE TASK ASKS — you looked, the affordance is not there, and no sequence of these verbs could produce it: a storefront with no way to rank by rating, a placed order whose address a customer cannot edit, a bulk operation the admin console does not expose, a repository you cannot add members to. finish with status action_not_allowed_error — or permission_denied_error when the site actively refuses YOU rather than simply lacking the control — citing the refs that show the absent or refusing affordance; the answer may then be empty. This is a finding about the SITE, exactly as not_found_error is a finding about the world, and it is scored as a correct answer. It is emphatically NOT a give_up: give_up says you failed, this says the site does not do that.
  - YOU could not complete the task — blocked, out of budget, lost, unable to evidence any progress: call give_up with a one-sentence reason. This is a statement about you. Never report it as a finish, never put "not_found_error" in the answer field, and never invent a placeholder like [] or None.
    give_up is the LAST resort, not the tidy way out, and it scores zero even when your work was right. Before calling it: if you have established what the task asked, finish with that; if you established that the thing is not there, that is not_found_error and it is a correct answer; if you established that the SITE has no control for what was asked, that is action_not_allowed_error and it is also a correct answer; if a finish was rejected, the rejection names what it wants — a rejected finish can be retried in the same episode, a give_up cannot be taken back. Only call it when none of those is true.
  Three of these four are correct answers and only give_up is a failure, so reach for it last. The zero-count and the does-not-exist pair are the easy two to confuse, and getting it wrong throws away a correct answer. Ask what the task asked you to RETURN. A number exists even over an empty set — zero is that number, and it is "success". A name, an entity, or a list of qualifying things cannot be produced from an empty set — when nothing qualifies, the finding is not_found_error, never "success" with []. Decide it by the SHAPE OF THE ANSWER, not by whether the inputs existed: if what you owe is a number, no emptiness upstream of it can turn it into an absence — you spent zero, so the total is zero, and that is "success".
- A TASK THAT ASKS YOU TO OPEN A PAGE IS FINISHED WHERE YOU STAND. When the task says to open, view, show, go to or bring up a page — a product, an order, a profile, a report — reaching it once is not doing it: the browser has to still be on that page when you finish, and every navigation you make afterwards replaces it as your answer. So do the checking BEFORE you arrive: pick the row from the list, confirm it is the right one from what the list already shows, then open it and finish there. If confirming genuinely requires opening something else, go back to the target page before you finish. Leaving to double-check a page you had already reached turns a correct answer into a wrong one.
- THE ADDRESS THE SITE GAVE YOU IS EVIDENCE; ONE YOU ASSEMBLE IS A GUESS. Reach a page by acting on the site's own control — the link, the sort, the filter — and let the page set its own address. Do not retype it, extend it or tidy it once you are standing there: a parameter you add to be more faithful to the task's wording makes a page the site never served, and it replaces the one you had. If you need a page nobody has handed you a link to, go and find the link: a link is a fact about the site, and a URL you compose is a hypothesis about it. THE ONE EXCEPTION, and it has a precondition you must satisfy first: a site's filter can be QUANTISED — the bound the task names may live in the site's parameter GRAMMAR while no control expresses it. A facet offering only coarse ranges cannot express a bound that falls inside one of them, and clicking the nearest range silently answers a different question. Before composing anything, ENUMERATE the whole facet and establish that no control reaches the bound — that is the precondition, and it is what separates this from guessing. Then read a facet link's own href to learn the grammar the site uses, substitute the task's value into it, and go. Afterwards the address is NOT your evidence: demand that the page's own applied-filter region name the bound you asked for, and that its item count reconcile against counts the site published before you composed anything. If either check fails, the server swallowed your parameter and you are standing somewhere you did not verify.
- BEFORE RANKING BY HAND, LOOK FOR THE CONTROL THAT RANKS. A list's own sort, filter, period or date-range control makes the site compute the answer and puts it in the first row; a report view or a contributors/statistics page may have computed it already. Reach for that first, and only enumerate when the page offers nothing. Sampling a few candidates and comparing them is not an extremum — it is a guess with arithmetic attached.
- AN EXTREMUM IS A PROCEDURE, NOT A JUDGEMENT. "closest", "cheapest", "nearest", "most recent", "largest" — before you answer one, do these in order and say which step you are on: (1) FIX THE REFERENCE POINT the task names and read back what the site resolved it to; (2) ENUMERATE the candidate set, consuming every continuation until nothing is withheld; (3) MEASURE or read the ranking value for EVERY candidate, not the two that look right; (4) TAKE THE MINIMUM, and if several tie the answer is all of them. Search order is not ranking order — the correct answer is not always near the top of the list. For an extremum over a category (cheapest X, most expensive X), walk the sorted results until the first item that IS an X — a page of non-matching items is not an answer. "The top N X" names the SET of the N highest-ranked, never the one ranked Nth: all N of them are the target, and a task that says to act on them means act on every one. And when the target is "all X by someone" or "all X in something", the population comes first: follow every continuation until nothing is withheld, count what you have, act on all of it, then count again — stopping at the first match, or at the end of the first page, silently answers a smaller question.
- THE PAGE OVERVIEW IS NOT AN INVENTORY OF THE PAGE. A read with no target returns announcements, regions, headings, collections and controls. Ordinary paragraph text belonging to none of those — a computed result, a total, a status sentence, a distance or a duration readout — is simply ABSENT from it, and the envelope will not report it as withheld, because it was never a candidate. So never conclude from an overview that an act produced nothing, or that a value is not on the page. When you expect the page to state something in prose, find it by a word you expect it to contain, or read the region that should hold it; one find by label reaches it in a single call. This applies only when what you want is free-standing TEXT — the structure the overview does list, it lists completely and with accounting.
- AN ACT'S VERDICT DESCRIBES WHAT THE RUNTIME COULD OBSERVE, NOT WHAT HAPPENED. no_observable_change_yet and unknown are not evidence that your act failed — an act can mutate the page substantially and still report one of these. And a verified verdict names the one transition it measured, which can be narrower than what you intended — a click that only collapsed a menu, or a submit whose delta captured a spinner while the real submission was still in flight, both report verified truthfully and prove nothing about the thing you wanted. So after an act you are going to DEPEND on — a commit, a state change you will cite, a navigation your answer rests on — confirm it independently: re-find the control and read its state back, or read the region that should have changed, and let that reading decide your next move. This does not apply to each fill inside a batch, it never makes contradicted into success, and it never licenses citing an act you have not settled. The cost it prevents is not a wasted call but a whole route abandoned as impossible when it had in fact worked.
- A FORM'S PRE-SET STATE IS PART OF THE FORM, NOT THE BACKGROUND. Before you submit anything, read the panel you are about to submit. Radios, checkboxes and selects commonly arrive already chosen, and the pre-set choice is frequently NOT the one the task wants — a commit panel that defaults to a new branch and an attached review request, a visibility that defaults to public, a delivery option already ticked. Submitting without reading them looks identical to deciding, and every later step still reports success while the thing the task asked for never happened. Read the controls, set the ones that are wrong, and confirm the flip by reading them back before you commit.
- A PRE-CHECKED MODE CONTROL DECLARES A COMPANION FIELD YOU MUST FILL. When a form offers modes — a kind of post, a type of entry, a method — and one arrives already selected, the field that mode governs is required even when the instruction never mentions it. The instruction names the field that is AMBIGUOUS; every other field of the submission is dictated by the form's own pre-set state plus the subject the task named. Leaving a declared field empty to avoid adding anything extra is backwards: it submits a different kind of record than the one the form said it was making.
- TRANSCRIBE A POPULATION AS THE SITE HOLDS IT, IN ITS OWN ORDER, REPEATS INCLUDED. When asked to list, copy or count what a page holds, do not tidy it: if the site shows two entries with the same text, write both. Deduplicating answers a neater question than the one you were asked, exactly as counting one page instead of the whole set does. THE PRECONDITION THAT BOUNDS THIS: keep a repeat only when it is a DISTINCT RECORD — two rows with the same words but different links, ids or authors. When the same row appears twice because you paged over it twice, or because a pager re-served rows you had already seen, that is one record seen twice and it belongs in your answer once. Compare the rows' own links or identifiers, not their text, to tell the two cases apart.
- WHEN A FORM DERIVES A SECOND FIELD FROM WHAT YOU TYPED, READ THE DERIVED VALUE BACK BEFORE COMMITTING. A name that generates an identifier, a title that generates a path or slug, a label that generates a key — the site rewrites your text under its own rules, lowercasing it, replacing punctuation or truncating it, and the derived field is frequently the one that matters. It is a separate control and it is readable: find it and read it before you submit, and if the task named the identifier itself rather than the display name, set the derived field explicitly rather than trusting what was generated.
- WHEN TWO FEATURES COULD BOTH SATISFY A REQUEST, THE SCOPE WORDS DECIDE. Sites commonly offer a rule that applies to a catalogue and a rule that applies at checkout, a filter that narrows a list and a report that aggregates one, a setting per record and the same setting globally. The task's scope words are the discriminator: "on all products" or "for every customer" names the broad one; a condition about a basket, an order, or a threshold names the one that acts at that moment. Reaching the right end state through the wrong feature writes a different record, and every step of it looks correct.
- Site knowledge may arrive during the run, as a block fenced
  <site_knowledge origin="..."> ... </site_knowledge>. It is notes about that
  exact deployment, gathered by exploring it earlier: what its controls are
  really named, which role they carry, how its pages are laid out and paginated.
  It is a map, not a claim about what is on screen now — use it to choose where
  to look and what to call things, and let the page win wherever the two
  disagree. Several may arrive on a task that spans sites; each applies ONLY to
  the origin it names, and the url in every response tells you which origin you
  are on. It is not page content and not part of the task.
- Respond only with tool calls.`;

// EXPERIMENT ONLY (docs/plans/read-screenshot-experiment.md), off by default.
// One variable drives all three surfaces — the attachment, the read tool's
// description, and this clause — because arms that differ in more than one thing
// measure nothing. Read once at import, like DEBUG_SCREENSHOTS and LOG_PROMPT:
// an episode is a process, so per-process is per-episode.
const READ_SCREENSHOT = process.env['WIR_READ_SCREENSHOT'] === '1';

// EXPERIMENT (exp/site-skills), off by default. Site knowledge for the origin
// the episode is on, appended ONCE when that origin is first observed.
//
// APPENDED, never merged into SYSTEM_PROMPT. Measured 2026-08-07: 95.4% of input
// tokens are served from the prefix cache at 1/50th price, and that holds only
// because the transcript grows at the end. Putting site text in the system
// prompt would change the prefix and re-bill everything after it — $0.06 -> $3.00
// on one suite — as well as making every episode pay for every site.
//
// The file is data, not code, and it lives in agent/, never core/: what the
// runtime does must not depend on which site it is looking at.
const SITE_SKILLS = process.env['WIR_SITE_SKILLS'] === '1';
// Resolved against the SOURCE tree, not the bundle: tsc emits .js and leaves
// .md behind, so `./skills/` next to the compiled loop is always empty. Found
// the hard way — the first live run injected nothing and said nothing about it.
const skillsDir = new URL('../../agent/skills/', import.meta.url);
const skillsSeen = new Set<string>();

/** `http://localhost:8023` -> `localhost_8023.md`. Origin only: a skill that
 *  varied by path would be a per-page script, which is a different and much
 *  worse idea. */
function siteSkill(rawUrl: string): string | null {
  if (!SITE_SKILLS) return null;
  let origin: string;
  try { origin = new URL(rawUrl).origin; } catch { return null; }
  if (skillsSeen.has(origin)) return null;
  const slug = origin.replace(/^https?:\/\//, '').replace(/[^a-z0-9]+/gi, '_');
  let text: string;
  try { text = readFileSync(new URL(`${slug}.md`, skillsDir), 'utf8'); }
  catch { skillsSeen.add(origin); return null; }   // no skill for this origin
  skillsSeen.add(origin);
  log(`site skill loaded for ${origin} (${text.split('\n').length} lines)`);
  return text;
}

// TWO MID-EPISODE NUDGES, on by default, each switchable alone so an arm can
// attribute its effect.
//
// Everything else this loop injects fires at the END of an episode — the mutate
// ledger, the skeptical evaluator, the give_up bounce, FORCE_FINISH. The
// 106-episode human-driver sweep (debug/runs/human-reddit/NOTES.md: 87 pass, 0
// runtime-owned losses) put the recoverable failures upstream of all of it. Its
// 14 driver-owned losses are two shapes and only two: settling on an answer
// before the population was enumerated (625, 626, 721, 735, and the forum-choice
// family 600/601/604/608/609), and concluding from a route never tried (722, 725,
// 617). This agent's own history has the same shape — docs/plans/fewer-misses.md
// measures 202 zero-result finds (36% of all finds) and 36 navigate rejections
// where every one was the same refused construction.
//
// Neither nudge is a capability and neither can decide anything. Each re-states,
// at the one moment it matters, something the episode already holds: a rejection
// the runtime already explained, or criteria the model itself already wrote.
const REPEAT_NUDGE = process.env['WIR_REPEAT_NUDGE'] !== '0';
const BUDGET_CHECKPOINT = process.env['WIR_BUDGET_CHECKPOINT'] !== '0';

// Staleness is stated once here rather than deleting old images, because the
// transcript is append-only and must stay that way: the prefix cache serves 95%
// of input tokens at 1/50th price, and mutating any earlier message makes every
// token after it a miss forever. The model already copes with stale read results
// for exactly this reason — each carries its epoch.
const SCREENSHOT_RULE = `
- After each read, a picture of the visible page is attached. It shows only what fits on screen, while read covers the whole page — the structure is the authority and the picture is a hint about layout, salience and grouping. Each is labelled with its read number and page epoch: only the LAST one shows the page as it is now; earlier ones are history, and acting on one is acting on a page that has moved.`;

const SYSTEM_PROMPT = READ_SCREENSHOT ? BASE_SYSTEM_PROMPT + SCREENSHOT_RULE : BASE_SYSTEM_PROMPT;

// Naming give_up here is the point: the recorded misuse of not_found_error
// happened on the forced final call, where this message demanded a finish and
// offered no other way out.
const FORCE_FINISH =
  'Budget is exhausted. Call finish now with your best answer and the evidence refs ' +
  'you have already observed. If you genuinely could not complete the task, call ' +
  'give_up with the reason instead — that is the honest outcome and it is scored ' +
  'as one. Do not call any other tool.';

// THE COMPLETION CONTRACT (RETRIEVE/NAVIGATE only): one extra exchange before
// the first tool turn, in which the model states what a complete answer to this
// task requires. Motive, measured: 8 of 10 agent losses were scope decisions
// made with COMPLETE data in hand — nothing partial, so the mechanical scope
// gate stayed silent by design. Criteria written down before the page is seen
// are criteria the finish can later be confronted with; the reply lives in the
// append-only transcript, where the prefix cache makes it nearly free on every
// later call. Wording is deliberately generic retrieval-task language — criteria
// steer models, and a site-shaped word here would be benchmark-shaped code.
const COMPLETION_CONTRACT = [
  'Before your first tool call: state what a COMPLETE answer to this task',
  'requires. At most 4 short bullets, plain text, no tool calls this turn:',
  '- the population of things the question is about, and where its boundary is',
  '- whether an ordering or extremum must be OBSERVED rather than assumed',
  '- what would distinguish "the answer is zero or empty" from "the thing does',
  '  not exist here" — and which status EACH side takes: a zero-valued count or',
  '  total is status "success" with 0; a name, entity, or list with nothing qualifying is',
  '  not_found_error, never "success" with an empty array',
  '- any exact-form requirement on the answer (units, casing, shape) — a declared',
  '  shape describes the FORM of an entry, never which entries belong: an entity',
  '  that qualifies but leaves one field unknown is still part of the answer',
  'These are your completion criteria. Gather evidence against them.',
].join('\n');

// Core's five verbs plus this controller's own give_up.
const TOOLS: ToolDefinition[] = [
  ...toolDefinitionsFor({ readScreenshot: READ_SCREENSHOT }), GIVE_UP_TOOL,
];

function log(message: string): void {
  process.stderr.write(`[loop] ${message}\n`);
}

function describe(error: unknown): string {
  return error instanceof Error ? (error.stack ?? error.message) : String(error);
}

function record(traj: WriteStream, entry: Record<string, unknown>): void {
  traj.write(`${JSON.stringify(entry)}\n`);
}

// PROVIDER PAYLOADS — what the model ACTUALLY received, one line per call.
//
// trajectory.jsonl cannot answer this and never could: wirDispatch writes its
// record BEFORE the payload is assembled, so likelyTargets, the site-skill
// block, the screenshot attachment and the repeat-rejection escalation are all
// appended afterwards. langfuse-trace.json is a {traceId, runId} POINTER by
// design — payloads travel to Langfuse Cloud over OTel and never touch disk,
// and fetch_trace.mjs truncates input to 1200 chars. WIR_LOG_PROMPT=1 writes
// the real thing to stderr, but covers only the main loop and has never once
// been switched on across 23 recorded studies.
//
// ALWAYS ON, deliberately. The defect being fixed is "the capability existed
// and nobody enabled it"; an opt-in flag reproduces exactly that. Delta form
// keeps it O(n) — ~100-120KB beside a 77KB trajectory, in study directories
// already running 165-232MB of trace.zip and network.har.
//
// Verbatim, not redacted: agent/telemetry/redact.ts settles this — redaction is
// the telemetry boundary only, the local debug plane stays lossless, and
// benchmark-results/ is gitignored.
let payloads: WriteStream | null = null;
let payloadCursor = 0;

/**
 * WHICH BUILD RAN THIS EPISODE. Not which commit existed when it ran — those are
 * different facts, and conflating them produced three separate wrong analyses in
 * one day: code sitting in the working tree at build time was declared absent
 * because its COMMIT timestamp fell after the build's mtime.
 *
 * buildMtime is the authority (it is the artifact that actually executed);
 * headSha is a convenience, and when HEAD has moved since the build the two
 * disagreeing is itself the signal. Read from .git directly rather than by
 * spawning git — the agent is a bare stdio process and must not shell out.
 */
function buildIdentity(): { buildMtime: string | null; headSha: string | null } {
  let buildMtime: string | null = null;
  let headSha: string | null = null;
  try {
    buildMtime = statSync(fileURLToPath(import.meta.url)).mtime.toISOString();
  } catch { /* non-fatal: an artifact field, never the episode */ }
  try {
    const root = `${dirname(dirname(fileURLToPath(import.meta.url)))}/..`;
    const head = readFileSync(`${root}/.git/HEAD`, 'utf8').trim();
    headSha = head.startsWith('ref: ')
      ? readFileSync(`${root}/.git/${head.slice(5)}`, 'utf8').trim()
      : head;
  } catch { /* a checkout without .git is legitimate */ }
  return { buildMtime, headSha };
}

/**
 * `whole` for a transcript that is not the episode's — the finish evaluator
 * builds its own two-message one. Otherwise only the messages appended since
 * the previous call, so concatenation reconstructs every payload exactly
 * without O(n^2) volume.
 */
function recordPayload(site: string, call: number, msgs: readonly ChatMessage[],
                       whole = false): void {
  if (payloads === null) return;
  const fresh = (whole ? [...msgs] : msgs.slice(payloadCursor))
    .map(m => ({ ...m, content: stripImageParts(m.content) }));
  record(payloads, { kind: 'payload', site, call, appended: fresh.length,
    total: whole ? fresh.length : msgs.length, messages: fresh });
  if (!whole) payloadCursor = msgs.length;
}

// Trajectory is the debug plane: lossless, always (docs/lessons.md preservation
// invariant 6 — no byte-prefix truncation in the audit trail).
function summarize(response: Record<string, unknown>): unknown {
  return response;
}

function evaluatorEnvelope(
  taskType: ExpectedAction, accepted: boolean, answer: string,
  finishStatus?: string | null,
): Record<string, unknown> {
  // Mechanical mapping onto the official Status enum, never a content judgment
  // (ADR-003). not_found_error passes through as NOT_FOUND_ERROR with null data
  // (earned by tasks 22/24 — the model had no channel to say "it does not
  // exist"). Non-accepted episodes map to UNKNOWN_ERROR: the previous 'FAILURE'
  // is not a member of the official enum at all (latent defect, same analysis).
  // All three observation-bar statuses, not just the first one wired. The enum
  // was widened in core/toolschemas.ts and core/session.ts and pinned by
  // test/finish-observation-statuses.test.ts, but this mapping was left behind,
  // so an ACCEPTED finish carrying action_not_allowed_error or
  // permission_denied_error went out as SUCCESS with the answer as
  // retrieved_data — the model had the vocabulary and the gate honoured it, and
  // then the envelope threw it away. Measured: 9 episodes whose grader demanded
  // exactly these two. Every member below is verified present in the official
  // Status enum (webarena_verified/types/agent_response.py:46-51).
  const OBSERVATION_STATUS_OUT: Record<string, string> = {
    not_found_error: 'NOT_FOUND_ERROR',
    action_not_allowed_error: 'ACTION_NOT_ALLOWED_ERROR',
    permission_denied_error: 'PERMISSION_DENIED_ERROR',
  };
  const observed = typeof finishStatus === 'string'
    ? OBSERVATION_STATUS_OUT[finishStatus] : undefined;
  if (accepted && observed !== undefined) {
    return { task_type: taskType, status: observed, retrieved_data: null };
  }
  // If the model produced JSON, hand the evaluator the structure, not a string
  // encoding of it. Proven: task 31 attempt 1 had the exactly-correct answer and
  // scored 0 on the string-vs-array shape alone.
  let data: unknown = answer === '' ? null : answer;
  if (typeof data === 'string') {
    const t = data.trim();
    if (t.startsWith('{') || t.startsWith('[')) {
      try { data = JSON.parse(t); } catch { /* keep the string */ }
    }
  }
  return {
    task_type: taskType,
    status: accepted ? 'SUCCESS' : 'UNKNOWN_ERROR',
    retrieved_data: data,
  };
}

// Debug-run mode (WIR_DEBUG_SCREENSHOTS=1): per-step PNGs + trace filmstrip land in
// the attempt directory, for humans only — nothing here ever reaches the model.
const DEBUG_SCREENSHOTS = process.env['WIR_DEBUG_SCREENSHOTS'] === '1';

// WIR_LOG_PROMPT=1: print the literal provider payload to stderr (the attempt's
// stderr.log) — the attempt-6 scar made this a named capability: verify what the
// model actually received. Call 1 logs the full message array (system + task);
// later calls log only the messages appended since the previous call, so the
// concatenation reconstructs every payload without O(n²) log volume.
const LOG_PROMPT = process.env['WIR_LOG_PROMPT'] === '1';

// Cost is computed only from operator-supplied rates (USD per million tokens) —
// a hardcoded price table would be an unverifiable constant. Absent rates leave
// estimatedCost null: unknown is not zero.
function priceRate(name: string): number | null {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return null;
  const value = Number(raw);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function estimateCost(metrics: Metrics): number | null {
  const input = priceRate('WIR_PRICE_INPUT_PER_MTOK');
  const output = priceRate('WIR_PRICE_OUTPUT_PER_MTOK');
  if (input === null || output === null) return null;
  const cachedRate = priceRate('WIR_PRICE_CACHED_INPUT_PER_MTOK') ?? input;
  const cached = metrics.cachedInputTokens ?? 0;
  return (
    (metrics.inputTokens - cached) * input + cached * cachedRate +
    metrics.outputTokens * output
  ) / 1_000_000;
}

async function startSession(request: AgentRequest): Promise<WirSession> {
  try {
    if (request.browser.cdpEndpoint !== undefined) {
      const session = await WirSession.attach({
        cdpEndpoint: request.browser.cdpEndpoint,
        expectedAction: request.task.expectedAction,
      });
      session.noteInstruction(request.task.instruction);
      await session.goto(request.task.startUrl);
      return session;
    }
    const session = await WirSession.start({
      headless: request.browser.headless,
      expectedAction: request.task.expectedAction,
      // The instruction feeds ONE ledger in core: a typed value the instruction
      // carries was given to the model, not invented (core/session.ts
      // GateEligibleAct.unseen). Core never compares it to an answer.
      instruction: request.task.instruction,
      storageStatePath: request.browser.storageStatePath,
      // Per-origin contexts are a runner declaration; passed through untouched.
      ...(request.browser.storageStates
        ? { storageStates: request.browser.storageStates } : {}),
      // Same producer gap the originPolicy spread below closed: core fences
      // uploads behind a runner-declared directory (core/act.ts resolveUpload),
      // but until this line nothing declared one, so every model `upload` was
      // invalid_args no matter the page or the file.
      ...(request.browser.uploadDir
        ? { uploadDir: request.browser.uploadDir } : {}),
      harPath: request.browser.harPath,
      tracePath: request.browser.tracePath,
      ...(request.browser.finalStatePath
        ? { finalStatePath: request.browser.finalStatePath } : {}),
      debugScreenshots: DEBUG_SCREENSHOTS,
      knownUrls: request.task.startUrls ?? [request.task.startUrl],
      // Without this the policy was unreachable from the agent process: core
      // accepted it, nothing passed it, and every episode ran 'declared'. An
      // open-web adapter could not opt in at all.
      ...(request.authority.originPolicy
        ? { originPolicy: request.authority.originPolicy } : {}),
    });
    await session.goto(request.task.startUrl);
    return session;
  } catch (error) {
    throw new BrowserError(describe(error));
  }
}

function initScreenshotsDir(trajectoryPath: string): void {
  if (!DEBUG_SCREENSHOTS) return;
  screenshotsDir = `${dirname(trajectoryPath)}/screenshots`;
  mkdirSync(screenshotsDir, { recursive: true });
  log(`debug run: per-step screenshots -> ${screenshotsDir}`);
}

// The screenshot is attached by THIS code, from a Buffer, into a typed content
// part. The model's whole output surface is tool_calls against a closed schema and
// no verb carries an image, so it can neither author one nor ask for one.
//
// Viewport-sized, no clip: the point is what the browser actually renders. JPEG for
// upload speed only — image tokens come from decoded dimensions, so the format does
// not change the bill (1280x900 ~= 1125 tokens, ~$0.00016 at mimo-v2.5 miss price,
// and cached at 1/50th of that on every later call).
async function readScreenshotMessage(
  session: WirSession, response: Record<string, unknown>, readNumber: number,
): Promise<ChatMessage | null> {
  let jpeg: Buffer;
  try {
    jpeg = await session.host.page.screenshot({ type: 'jpeg', quality: 70, fullPage: false });
  } catch (error) {
    // A failed capture must not end an episode that would otherwise have run:
    // the arm degrades to the baseline for this one call and says so.
    log(`read screenshot failed (non-fatal, no image attached): ${describe(error)}`);
    return null;
  }
  const epoch = String(response['documentEpoch'] ?? 'unknown');
  const parts: ContentPart[] = [
    { type: 'text', text: `Page after read ${readNumber} (epoch ${epoch}).` },
    { type: 'image_url', image_url: {
      url: `data:image/jpeg;base64,${jpeg.toString('base64')}`, detail: 'high' } },
  ];
  return { role: 'user', content: parts };
}

async function wirDispatch(
  session: WirSession, verbRequest: VerbRequest, metrics: Metrics, traj: WriteStream,
): Promise<Record<string, unknown>> {
  if (verbRequest.verb === 'read' && lastVerb === 'read') metrics.readAfterRead += 1;
  lastVerb = verbRequest.verb;
  metrics.wirCalls += 1;
  let response: Record<string, unknown>;
  const verbStartedAt = Date.now();
  try {
    response = await session.dispatch(verbRequest);
  } catch (error) {
    trace?.toolCall({ verb: verbRequest.verb, args: verbRequest, result: null,
      ms: Date.now() - verbStartedAt, error: describe(error) });
    throw new BrowserError(describe(error));
  }
  const ms = Date.now() - verbStartedAt;
  trace?.toolCall({ verb: verbRequest.verb, args: verbRequest, result: response, ms });
  if (verbRequest.verb === 'act' && response['outcome'] === 'delivered') {
    metrics.browserDeliveries += 1;
  }
  if (verbRequest.verb === 'find') {
    const population = response['population'] as { matched?: unknown } | undefined;
    if (population && typeof population.matched === 'number') {
      findPopulations.push({ query: JSON.stringify(verbRequest), matched: population.matched });
    }
  }
  // The bytes the model actually receives for this verb result — the observation
  // channel the projection budgets exist for.
  const responseBytes = Buffer.byteLength(JSON.stringify(response));
  metrics.observationBytes += responseBytes;
  record(traj, { kind: 'wir', ms, responseBytes, request: verbRequest, response: summarize(response) });
  if (DEBUG_SCREENSHOTS && screenshotsDir !== null) {
    const file = `${screenshotsDir}/step-${String(metrics.wirCalls).padStart(3, '0')}-${verbRequest.verb}.png`;
    session.host.page.screenshot({ path: file }).catch(e =>
      log(`screenshot failed (non-fatal): ${describe(e)}`));
  }
  return response;
}

let screenshotsDir: string | null = null;
let trace: EpisodeTrace | null = null;
// Previous verb, for the adjacent read→read count. Module-level like the two above:
// one episode per process.
let lastVerb: string | null = null;
let readCount = 0;
// What each find counted, in the order the model asked. Agent-side on purpose:
// `population` is a field of the result the model already received, so recording
// it duplicates no core vocabulary — unlike continuation identity, which is
// core's own pagination rule and lives there.
const findPopulations: { query: string; matched: number }[] = [];
// The model's own statement of what a complete answer requires, taken before the
// first tool turn (the completion contract, drive()). Module-level like the
// three above: one episode per process.
let contractText: string | null = null;
// Every rejected call this episode, keyed by (verb, arguments, rejection kind) —
// the repeat-rejection escalation's whole state, and the count is per signature,
// never global. Same module-level convention, same reason: one episode per process.
const rejectionCounts = new Map<string, number>();
let repeatNudges = 0;
// Set the moment the model reaches for `finish`, BEFORE any confrontation can
// bounce it: an attempted finish is an attempt whatever the gate did with it, and
// the budget checkpoint exists only for a model that has not tried at all.
let finishAttempted = false;
let budgetCheckpointFired = false;
// NEXT-CALL OFFERS (agent/offers.ts): the literal `until` / `expect` call,
// appended where the runtime's own result shows the pattern. Same module-level
// convention, same reason: one episode per process. actRecords keeps every
// delivered act's request and result by actRef so the MUTATE confrontation can
// build the `expect.sent` a cited act would have taken from that act's own
// receipt; filledValues is what the model itself put into fields, so the
// offered `fields` are the ones it meant and never the site's hidden inputs.
const offerState = createOfferState();
const actRecords = new Map<string, { args: Record<string, unknown>; response: Record<string, unknown> }>();
const filledValues = new Set<string>();

/** Evidence that an act changed ITS OWN TARGET and nothing else, so the refs the
 *  rest of the batch was written against still name the same nodes.
 *
 *  Deliberately a whitelist. Anything not named here — a navigation, a
 *  client-side re-render (`dom_mutated`), a dialog, a popup, a download, a
 *  committed request — may have moved the document out from under the remaining
 *  calls, and those still end the turn. A rejection is not in this set and does
 *  not need to be: nothing happened, so the batch continues. */
const LOCAL_EFFECT_EVIDENCE = new Set([
  'value_set', 'value_mismatch', 'option_selected', 'selection_mismatch',
  'text_typed', 'text_edited', 'selection_changed', 'file_attached',
  'target_state_changed', 'no_observable_change_yet',
  'scrolled', 'scrolled_no_new_content',
]);

/** Did this act's effect stay on its own target? Unparseable or unfamiliar
 *  answers count as "moved" — the conservative direction. */
function effectStayedLocal(content: string): boolean {
  try {
    const parsed = JSON.parse(content) as
      { rejected?: unknown; effect?: { evidence?: string } };
    if (parsed.rejected !== undefined) return true;   // refused: the page did not move
    const evidence = parsed.effect?.evidence;
    return typeof evidence === 'string' && LOCAL_EFFECT_EVIDENCE.has(evidence);
  } catch {
    return false;
  }
}

interface ToolOutcome {
  content: string; finishAccepted: boolean; answer: string;
  finishStatus: string | null;
  gaveUp?: { reason: string };
  /** An extra message the loop appends AFTER the tool result. A tool-role message
   *  cannot carry an image on this wire, so the screenshot rides as its own
   *  user-role message — appended, never replacing anything. */
  attachment?: ChatMessage;
}

// A tool call the loop refuses before it ever reaches WIR. record() fires only
// inside wirDispatch, so these — garbled JSON arguments, an unknown tool name, a
// budget refusal — reached the model and left NO trace: the debug plane lost
// exactly the calls that went wrong (lessons.md preservation invariant 6).
// rawArguments is the model's string verbatim, unparsed and unbounded; the
// trajectory is the lossless plane, and a garbled argument is only diagnosable
// in the form the model actually emitted.
function rejection(traj: WriteStream, call: ToolCall, reason: string): ToolOutcome {
  record(traj, {
    kind: 'rejected', name: call.function.name,
    rawArguments: call.function.arguments, reason,
  });
  return {
    content: JSON.stringify({ rejected: { kind: 'invalid_args', reason } }),
    finishAccepted: false,
    answer: '',
    finishStatus: null,
  };
}

// NUDGE 1 — REPEAT-REJECTION ESCALATION.
//
// A rejection already carries a reason and usually a literal repair, and the
// model still re-sends the identical call. The measured populations are exactly
// this shape: all 36 navigate `invalid_args` rejections were the same refused
// construction (an absolute URL the observed-URL closure will never admit), and
// 84 of 86 `unknown_ref` rejections cite a ref the runtime itself delivered
// (docs/plans/fewer-misses.md A3, A4). Neither is repaired by re-sending; both
// look, from inside the transcript, like a call that has not been answered yet.
//
// So on the SECOND rejection of the same call with the same kind, say so once,
// in the runtime's own words. This invents no advice: when the rejection carried
// a repair, that repair is what travels — the runtime already computed the
// literal valid continuation and the escalation just points at it again.

/** How many DISTINCT calls may be escalated in one episode. */
const REPEAT_NUDGE_LIMIT = 3;
// Bounded for the same reason as the repeat nudge: advice that arrives on every
// one of ~30 unmoved turns stops being advice and becomes noise the model learns
// to skip, and it costs prefix-cache-priced tokens on each one.
const BATCH_NUDGE_LIMIT = 3;

/** (verb, arguments, rejection kind), canonical. Keys are sorted so the same call
 *  re-emitted with its keys in another order is still the same call — which is
 *  what "you have already tried this" has to mean. Arguments that do not parse are
 *  their own signature verbatim: a garbled string repeated is still a repeat. */
function callSignature(name: string, rawArguments: string, kind: string): string {
  let args = rawArguments;
  try {
    const parsed = JSON.parse(rawArguments || '{}') as Record<string, unknown>;
    args = JSON.stringify(Object.fromEntries(
      Object.entries(parsed).sort(([a], [b]) => a.localeCompare(b))));
  } catch { /* keep the raw string */ }
  return `${name} ${args} ${kind}`;
}

/** The rejection inside a tool result, or null when the call was not rejected.
 *  Covers both sources on purpose — core's typed rejections and the loop's own
 *  pre-dispatch refusals share this envelope, and a model repeating a garbled
 *  argument is drifting exactly as hard as one repeating a dead ref. */
function rejectionOf(content: string): { kind: string; repair: string | null } | null {
  try {
    const parsed = JSON.parse(content) as
      { rejected?: { kind?: unknown; repair?: unknown } };
    const r = parsed.rejected;
    if (r === undefined || typeof r.kind !== 'string') return null;
    return { kind: r.kind, repair: typeof r.repair === 'string' ? r.repair : null };
  } catch { return null; }
}

/** One line to append to this tool result, or null. Fires on the second rejection
 *  of a signature and never again for that signature, at most REPEAT_NUDGE_LIMIT
 *  times per episode; a call that is not rejected is never touched. */
function repeatEscalation(call: ToolCall, content: string, traj: WriteStream): string | null {
  if (!REPEAT_NUDGE) return null;
  // `finish` is deliberately outside this. The gate carries its own repeat
  // detector already — doFinish compares the query hash and appends "resubmitted
  // unchanged — change the finish or gather evidence" (core/session.ts) — and
  // "change the route, not the arguments" is the wrong advice for a finish, where
  // changing the arguments is precisely the repair.
  if (call.function.name === 'finish') return null;
  const rejected = rejectionOf(content);
  if (rejected === null) return null;
  const signature = callSignature(call.function.name, call.function.arguments, rejected.kind);
  const seen = (rejectionCounts.get(signature) ?? 0) + 1;
  rejectionCounts.set(signature, seen);
  if (seen !== 2 || repeatNudges >= REPEAT_NUDGE_LIMIT) return null;
  repeatNudges += 1;
  const line =
    `This exact ${call.function.name} call has now been rejected twice, both times `
    + `with ${rejected.kind}. Sending it a third time is refused a third time — `
    + 'change the route, not the arguments.'
    + (rejected.repair !== null ? `\nThe runtime's repair for it: ${rejected.repair}` : '');
  record(traj, {
    kind: 'repeat_rejection_nudge', verb: call.function.name,
    rejectionKind: rejected.kind, occurrences: seen, nudge: repeatNudges,
    limit: REPEAT_NUDGE_LIMIT, arguments: call.function.arguments,
    repair: rejected.repair, chars: line.length,
  });
  return line;
}

// NUDGE 2 — THE BUDGET CHECKPOINT.
//
// The completion contract is written before the page is seen and then read only
// at the end, by the skeptical evaluator, on a finish that has already been
// composed. The sweep's largest driver-owned class settles well before that: an
// answer chosen from a partial enumeration, with calls still in hand. This puts
// the model's own criteria back in front of it while there is still budget to act
// on them, and it says nothing the model did not write itself.
function budgetCheckpointMessage(contract: string, calls: number, max: number): string {
  return [
    `${calls} of your ${max} model calls are spent and you have not attempted a finish `
    + 'yet — these are the completion criteria you wrote before you saw the page, and '
    + 'nothing has been submitted against them:',
    '',
    contract,
  ].join('\n');
}

// The MUTATE gate proves that *a* mutation happened — never that the whole task
// did. gitlab 743 and 747 both finished having done part of a multi-step
// instruction: the gate correctly accepted (a project WAS created) and the
// official evaluator scored 0 on the member-add criteria. Nothing in the runtime
// can know the task had three steps — that lives in the evaluator's criteria,
// which the runtime must never read.
//
// So: surface, do not judge. On a MUTATE episode the first finish attempt is
// bounced once with the mechanical ledger of what the runtime actually proved,
// and the model re-reads its own instruction against it. There is NO comparison
// of the answer to the instruction anywhere here (ADR-003 untouched) — the
// ledger is facts the model already had a right to, collected in one place at
// the one moment it matters.
function provenActs(session: WirSession): string {
  const ledger = session.gateEligibleActs();
  return ledger.length === 0
    ? '  (none — no act has proven a site-side change yet)'
    : ledger.map(a => `  ${a.actRef}  ${a.action}  ${a.evidence}${a.sent ? `  sent: ${a.sent}` : ''}`
        + `${a.expectation ? `  declared: ${a.expectation}` : ''}`
        // Beside `sent:`, the delivery fact core recorded at act time: which of
        // those values the page had not shown before the model typed them.
        // Restated, never judged — the finish citing this act is still accepted.
        + `${a.unseen ? `  values the page never showed you: ${a.unseen.join(', ')}` : ''}`
        + `${a.unseenUnchecked ? `  (${a.unseenUnchecked})` : ''}`).join('\n');
}

// THE DECLARATION A CITED ACT DID NOT MAKE, as the literal call. "Declare what
// you expect an act to send" was advice, and arm 4 used expect zero times in 20
// episodes; this is the same sentence as the call the model could have sent,
// built from the cited act's own receipt (agent/offers.ts sentExpectCall). One
// line per cited act that declared nothing; an act whose receipt carried no
// request has nothing to build from and is told that instead.
function undeclaredCitedActs(rawArguments: string): string[] {
  const lines: string[] = [];
  for (const ref of evidenceRefsOf(rawArguments)) {
    const rec = actRecords.get(ref);
    if (rec === undefined) continue;
    if (rec.args['expect'] !== undefined && rec.args['expect'] !== null) continue;
    const literal = sentExpectCall(rec.args, rec.response, filledValues);
    lines.push(literal === null
      ? `  ${ref} declared nothing and its receipt holds no request — declare expect.sent on the act that submits`
      : `  ${ref} declared nothing; the call that would have proved the intent: ${literal}`);
  }
  return lines;
}

export function mutateConfrontation(session: WirSession, rawArguments: string): string {
  const undeclared = undeclaredCitedActs(rawArguments);
  return [
    'Before this finish is submitted, confirm what the runtime actually proved.',
    '',
    'Gate-eligible verified acts this episode — the only acts that can prove the',
    'site changed:',
    provenActs(session),
    ...(undeclared.length > 0 ? ['', 'Cited acts that declared no expectation:', ...undeclared] : []),
    '',
    // One sentence, only when an act carries the fact. Task 521 three times:
    // the model typed user@example.com (or its operator's address) into the
    // newsletter box while the customer's own sat on My Account, unread.
    ...(session.gateEligibleActs().some(a => a.unseen !== undefined)
      ? ['A value the page never showed came from you; if the task meant the',
         "account's own, read where the account states it.", '']
      : []),
    'Re-read the task instruction and walk it step by step against that ledger.',
    'If a step of the task has no act here, that step probably did not happen —',
    'go and do it. If the ledger covers the whole task, call finish again to',
    'submit it. Declare what you expect an act to send or show; a held expectation',
    'is what a MUTATE finish should cite. This check happens once.',
  ].join('\n');
}

// A bounded list still accounts for what it left out — the same rule the verbs
// obey. Ranked by the mechanical magnitude so the biggest unread region is never
// the one that falls off the end.
const CONFRONT_LIST = 6;

function bounded(lines: string[], empty: string): string {
  if (lines.length === 0) return `  ${empty}`;
  const shown = lines.slice(0, CONFRONT_LIST);
  const left = lines.length - shown.length;
  return [...shown, ...(left > 0 ? [`  (+${left} more, all smaller)`] : [])].join('\n');
}

// The RETRIEVE/NAVIGATE analogue of the MUTATE ledger, and the same principle:
// surface, do not judge.
//
// Measured on the failed-10 study (51 trajectories): 30 of 41 failures ended with
// the agent reporting success while the official evaluator scored 0, at a median
// of 31 model calls against a cap of 60. Not budget deaths — the agent stopped
// early, confident and wrong. Task 68's expected answer holds two books; two
// independent runs answered with one and finished with calls to spare, one of
// them with a 37-child continuation still unopened on the page it was reading
// (benchmark-results/failed10-develop-1/task-68/attempt-1).
//
// Both facts below are the runtime's own accounting, already delivered to the
// model in results it received: pagination it was offered and never called back,
// and what its own finds counted. The answer is compared to nothing — that
// comparator is the matcher ADR-003 forbids, and correctness stays the official
// evaluator's alone.
function unreadOffers(session: WirSession): string {
  // ROUND-ROBIN ACROSS UNITS, not a global sort by count. The counts measure
  // different things — characters of a cut comment, items of an unopened list,
  // matches a find withheld — and sorting them together lets the largest unit
  // win every slot. Observed on a page with 30 long comments and a 60-item
  // list: all six lines were 419-character comment tails, and the 50-item list,
  // the 40 withheld controls and a 20-node census were evicted entirely. The
  // model would be sent to re-read comment endings while a list it had never
  // opened stayed invisible.
  //
  // Within a unit the largest still comes first; across units each takes a turn.
  // TRIVIAL RESIDUE IS NOT AN OBJECTION. A 56-character tail or a single
  // withheld item is interface noise present on almost every page, and the
  // evaluator's own brief already calls this class "grounds for nothing by
  // itself" — yet it has now bounced two CORRECT finishes on exactly that:
  // shopping-286 (a NAVIGATE finish made from the graded page, after which the
  // agent navigated away to re-check and lost the task) and map-378 (bounced on
  // 56 characters, passed unchanged on resubmission). Filtering here rather than
  // arguing with the model in the prompt: an objection it never sees is one it
  // cannot make. The thresholds are deliberately low — anything that could
  // plausibly hold an answer still travels.
  const SUBSTANTIVE = (o: ContinuationOffer): boolean =>
    o.unit === 'characters' ? o.withheldCount >= 200 : o.withheldCount >= 2;
  const byUnit = new Map<string, ContinuationOffer[]>();
  for (const o of session.unconsumedContinuations()) {
    if (!SUBSTANTIVE(o)) continue;
    byUnit.set(o.unit, [...(byUnit.get(o.unit) ?? []), o]);
  }
  for (const list of byUnit.values()) list.sort((a, b) => b.withheldCount - a.withheldCount);
  const queues = [...byUnit.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(e => e[1]);
  const unread: string[] = [];
  for (let round = 0; queues.some(q => q.length > round); round += 1) {
    for (const q of queues) {
      const o = q[round];
      if (o) unread.push(`  ${o.withheldCount} ${o.unit} withheld — ${o.call}`);
    }
  }
  return bounded(unread, '(none — nothing offered on this page is still unread)');
}

function findCounts(): string {
  const finds = [...findPopulations]
    .sort((a, b) => b.matched - a.matched)
    .map(f => `  matched ${f.matched} — ${f.query}`);
  return bounded(finds, '(none — you never used find this episode)');
}

/** The evidenceRefs the model just cited, from its own raw tool arguments. Never
 *  throws: a malformed finish is the runtime's to reject, not this helper's. */
function evidenceRefsOf(rawArguments: string): string[] {
  try {
    const parsed = JSON.parse(rawArguments) as { evidenceRefs?: unknown };
    return Array.isArray(parsed.evidenceRefs)
      ? parsed.evidenceRefs.filter((x): x is string => typeof x === 'string')
      : [];
  } catch { return []; }
}

// THE SKEPTICAL EVALUATOR, replacing the scope self-confrontation on
// RETRIEVE/NAVIGATE. Two predecessors, both measured short: the byte-ranked
// bounce fired on 98% of episodes and changed the answer in ~2% (n=57); the
// scope rebuild fired only when a cited population was PARTLY delivered — and
// 8 of 10 agent losses were scope decisions made with COMPLETE data in hand,
// so the trigger structurally could not see the loss class it was built for.
//
// So the first finish is now judged by a second model call on a FRESH
// conversation — never the episode transcript — carrying only runtime-known
// facts: the instruction, the declared answer shape, the model's own
// completion criteria, the pending finish, scopeFacts (including the
// unknowable disclosure), and unconsumed pagination. No page content beyond
// what those contain: an evaluator that read the page would be a second
// answerer, and the runtime still compares the answer to nothing (ADR-003) —
// the objection is advice to the model, never authority over the gate, and
// the gate's own rules are untouched.
//
// The stance is refutation because approval is the default failure mode of a
// model reviewing a model: asked "is this right?", it says yes. Asked "name
// the specific reason the scope is wrong", it must produce a concrete
// objection or approve — and an unparseable or errored verdict FAILS OPEN to
// approve, recorded as such, because a guard must never brick an episode.
const EVALUATOR_SYSTEM = [
  'You evaluate the final answer an agent is about to submit for a web',
  'information task. Your stance is refutation: find the specific reason the',
  "answer's SCOPE is wrong — the population it drew from is narrower or broader",
  'than what the question names, an ordering or extremum was assumed rather',
  'than observed, a zero-or-empty result was conflated with the thing not',
  "existing, or the answer's form does not match what was asked.",
  'You cannot see the page; judge only from the facts given. Approve only if',
  'you cannot name such a reason concretely.',
  'The facts arrive in two sections and they do not weigh the same. A CITED',
  'population only partly delivered is strong grounds for objection. Generic',
  'unconsumed pagination — unread character tails, withheld controls — is',
  'routine interface residue on every page, and grounds for nothing by itself:',
  'object to it only when the task gives a concrete reason that content holds',
  'the answer.',
  'Deliver the verdict by calling the verdict tool — the only tool offered and',
  'the only reply that is read. Prose is discarded.',
].join('\n');

// THE VERDICT ARRIVES THROUGH A FORCED TOOL, not as JSON-in-text.
// sweep17-integration measured the text transport broken on mimo: 4 of 7
// recorded verdicts unparseable, every one failing open. Three replied with a
// duplicate `finish` tool call — task-147/attempt-1/trajectory.jsonl record 47
// has content "" and arguments carrying the agent's own finish payload — and
// one with prose analysis (task-235 record 15). Root cause: the evaluator call
// passed the EPISODE tools, so a tool-reaching model had six wrong affordances
// and no right one; 147 was a missed save on the evaluator's exact target
// class. Now the call carries exactly this one tool and tool_choice forces it:
// the recorded mistake shape is no longer expressible.
const VERDICT_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: 'verdict',
    description:
      'Submit your evaluation. Set approve to true when you cannot name a ' +
      'concrete objection — omit objection entirely. Set approve to false when ' +
      'you can, and put that one specific, actionable sentence in objection; ' +
      'approve=false with no objection is invalid.',
    parameters: {
      type: 'object',
      properties: {
        approve: {
          type: 'boolean',
          description: 'true: the answer stands. false: it has a named scope defect.',
        },
        objection: {
          type: 'string',
          description: 'only when approve is false: the one specific, actionable sentence the agent will be shown',
        },
      },
      required: ['approve'],
      additionalProperties: false,
    },
  },
};

// Two fact sections, explicitly ranked, because the evaluator weighed them
// flat: task-22's rejection objected about "7467 and 6507 characters of unread
// text" — affordance-tail character offers, the known noise class — and forced
// a 27-call re-read of already-complete review data (sweep17-integration/
// task-22/attempt-1/trajectory.jsonl record 59). Scope facts on collections
// the finish CITES are the strong signal; generic unconsumed continuations
// are weak. Nothing is dropped — order and label, never remove.
function evaluatorBrief(
  task: AgentRequest['task'], session: WirSession,
  finish: { answer: string; status: string; refCount: number },
  facts: ReturnType<WirSession['scopeFacts']>,
): string {
  const factLines = facts === null
    ? '  (unknowable: no compiled graph at finish time, so the populations behind\n'
      + '  the cited refs could not be checked)'
    : facts.length === 0
      // TWO DIFFERENT SITUATIONS, and conflating them manufactured an objection
      // to correct answers. A page with no compiled collection at all — a product
      // detail page, a profile, a single record — has no population to have
      // missed, and saying "the cited refs belong to no compiled collection"
      // there reads as a shortfall in the same slot a real one appears. Every
      // "go to the page for X" episode ended that way. Where collections DO
      // exist and the refs sit outside them, that is worth saying plainly.
      ? (session.collectionCount() ?? 0) === 0
        ? '  (this page compiles no collection, so there is no population to account for)'
        : `  (the cited refs sit outside the ${session.collectionCount()} compiled `
          + 'collection(s) on this page — expected when the answer is a single '
          + 'record rather than a set, worth a second look when it is a set)'
      : facts.map(f =>
          `  population ${f.collection}: ${f.items} items exist, ${f.delivered} were `
          + `shown this episode, the answer cites ${f.cited}`
          + (f.continuation ? `\n    unread: ${f.continuation}` : '')).join('\n');
  return [
    `Task: ${task.instruction}`,
    task.resultsSchema
      ? `Declared answer shape (JSON Schema): ${JSON.stringify(task.resultsSchema)}`
      : 'No answer shape was declared.',
    '',
    'The agent stated these completion criteria before touching the page:',
    contractText ?? '  (none recorded)',
    '',
    `Pending finish: status ${finish.status}, ${finish.refCount} evidence refs cited, answer:`,
    finish.answer === '' ? '  (empty)' : `  ${finish.answer}`,
    '',
    'STRONG SIGNAL — what the runtime observed about the populations behind the',
    'refs the finish cites (delivered counts against what exists):',
    factLines,
    '',
    'WEAK SIGNAL — generic pagination the runtime offered this episode and the',
    'agent never consumed (each entry labeled with its unit):',
    unreadOffers(session),
  ].join('\n');
}

/** The verdict, or null when the reply is not one — the caller fails open.
 *  The forced tool call is the transport, so its arguments are checked first.
 *  Text JSON stays as a fallback for a provider that ignores tool_choice but
 *  still answers in the old shape; fence-stripping is the one liberty taken
 *  there, because models wrap JSON in markdown fences routinely, and rejecting
 *  that shape would turn a working evaluator into a permanent approver. */
// Shown when the evaluator replied but its verdict could not be read. Names the
// exact shape, because the recorded failures were argument-name guesses:
// {"decision":"accept"} and {"approve":true} were both sent in one sweep.
const VERDICT_UNPARSEABLE = [
  'Your finish was NOT submitted: the review step replied, but its verdict could',
  'not be read, and an unreadable verdict is not an approval.',
  '',
  'The verdict tool takes exactly one required argument:',
  '  approve: boolean   — true to let the answer stand, false to object',
  '  objection: string  — required ONLY when approve is false',
  '',
  'Answer the review turn with that tool and those argument names, then submit',
  'the finish again. A rejected finish can always be retried.',
].join('\n');

function evaluatorVerdict(message: ChatMessage):
    { approve: boolean; objection: string } | null {
  const call = (message.tool_calls ?? []).find(c => c.function.name === 'verdict');
  if (call) return verdictOf(call.function.arguments);
  const text = typeof message.content === 'string' ? message.content : '';
  return verdictOf(text.trim().replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, ''));
}

function verdictOf(json: string): { approve: boolean; objection: string } | null {
  try {
    const parsed = JSON.parse(json) as { approve?: unknown; objection?: unknown };
    if (parsed.approve === true) return { approve: true, objection: '' };
    if (parsed.approve === false && typeof parsed.objection === 'string'
        && parsed.objection.trim() !== '') {
      return { approve: false, objection: parsed.objection };
    }
  } catch { /* not a verdict */ }
  return null;
}

function objectionBounce(objection: string): string {
  return [
    'Before this finish is submitted, one objection to its scope:',
    '',
    `  ${objection}`,
    '',
    'If the objection is mistaken, call finish again and it will be submitted',
    'unchanged. If it is right, gather the missing evidence first. This check',
    'happens once.',
  ].join('\n');
}

// The same treatment for the other terminal tool, and it was missing.
//
// `finish` earned its confrontation because premature termination was the
// measured problem; the tool whose ONLY function is premature termination had no
// guard at all — one call, no gate, no ledger. Measured across the failed-10
// studies: 8 abstentions, and every reason is a claim the runtime's own
// accounting can speak to — "could not find the control", "the user does not
// exist in the member list" (failed10-proj/task-659/attempt-1), "the acts
// receive unknown verdicts" (failed10-mimo-1/task-612/attempt-1, at call 54 of
// 60 with six calls still in hand).
//
// Unread pagination shows on BOTH modes here, unlike the finish confrontation:
// "I could not find X" is exactly the claim an unopened continuation refutes,
// whatever the task type. Same principle as the other two — surface, do not
// judge; the reason is compared to nothing.
//
// This is one bounce, not a gate. give_up must stay reachable: it is the honest
// channel for "I could not do this", it maps to the scored agent_abstained
// outcome, and the caller clears this flag whenever the model has no turn left
// to spend — a model that means it says it twice and the episode ends.
function giveUpConfrontation(session: WirSession): string {
  return [
    'Before the episode ends in failure, confirm you have used what the runtime',
    'already handed you.',
    '',
    // Shown on EVERY mode, not just MUTATE. Measured over 46 give-ups: 40 held a
    // gate-eligible verified act at the moment they quit, and on a RETRIEVE or
    // NAVIGATE episode this bounce never once mentioned it — the model was asked
    // "have you used what you were handed" while the strongest thing it held was
    // withheld from the question.
    ...(session.gateEligibleActs().length > 0
      ? ['What the runtime has proved you already did — this is evidence you hold now:',
         provenActs(session), '']
      : []),
    'Pagination this page offered you and you never called back — content the',
    'runtime holds and you have not read:',
    unreadOffers(session),
    '',
    'What your own finds counted this episode:',
    findCounts(),
    '',
    'If anything above could still hold what you were missing, go and read it —',
    'a thing you never looked at is not a thing that is not there.',
    '',
    // THE THIRD DOOR. This bounce used to offer exactly two: look harder, or
    // quit and fail. It never said that an established absence is a correct
    // ANSWER, and the enum has carried the words for it since the status
    // widening. map-101 is the shape: the task asks for the nearest In-N-Out to
    // a Pittsburgh landmark, there is none, the grader expects not_found_error —
    // and the episode gave up, scoring UNKNOWN_ERROR for having found exactly
    // the right thing. The ledger-conditioned second bounce cannot help here
    // either: a read-only episode has no gate-eligible act, so that predicate is
    // structurally empty on every RETRIEVE of this kind.
    'But if you DID look and the thing is genuinely not there, that is not a',
    'failure and it is not a give_up: finish with status not_found_error, citing',
    'the refs where you looked. "It does not exist" is a correct answer and it is',
    'scored as one. The same goes for a permission wall or a control the site',
    'refuses you: those are permission_denied_error and action_not_allowed_error,',
    'and they are answers too. And if you have already established what the task',
    'asked, finish with that instead — a rejected finish can be retried, a',
    'give_up cannot be taken back.',
    '',
    'Only if none of those is true, call give_up again and the episode ends as a',
    'failure, which is then the honest outcome. This check happens once.',
  ].join('\n');
}

async function executeToolCall(
  session: WirSession, call: ToolCall, metrics: Metrics, traj: WriteStream,
  request: AgentRequest,
  confront: { finish: boolean; giveUp: boolean; giveUpSecond: boolean; verdictBounced: boolean },
): Promise<ToolOutcome> {
  const outcome = await dispatchToolCall(session, call, metrics, traj, request, confront);
  const escalation = repeatEscalation(call, outcome.content, traj);
  if (escalation === null) return outcome;
  // APPENDED INTO THE TOOL RESULT BEING BUILT — not pushed as a separate message.
  //
  // Both are append-only and both are identical for the prefix cache: this
  // message has not been sent yet, so nothing before it changes either way. The
  // tie is broken by the batch. A turn may carry several tool calls, and their
  // results are pushed one per call; a user-role message inserted between two of
  // them splits a tool block that must answer every tool_call_id of one assistant
  // turn. The attachment slot already takes that risk for the screenshot arm and
  // the site skill (both at most once per episode). A rejection nudge can fire
  // mid-batch, so it rides inside the result it is about, where it also reads as
  // what it is: more of the runtime's answer to THIS call.
  return { ...outcome, content: `${outcome.content}\n${escalation}` };
}


// ---------------------------------------------------------------- node ranker
// ADVISORY shortlist from a local cross-encoder (Ettin-17m, fine-tuned on
// 3,872 node choices from winning episodes). OFF unless WIR_NODE_RANKER is set.
//
// It ADDS a `likelyTargets` field and removes nothing: the full node list still
// reaches the model exactly as before, so "ranking may order, never remove"
// holds literally — this does not even reorder, it annotates.
//
// Measured offline on a template-held-out split: R@1 32, R@10 79 against a
// lexical-overlap baseline of R@1 7.6. It is a hint, not an oracle, and it is
// labelled as one in the payload so the model can disregard it.
const NODE_RANKER = process.env['WIR_NODE_RANKER'] ?? '';
function collectRefs(o: unknown, out: { ref: string; text: string }[], seen: Set<string>): void {
  if (Array.isArray(o)) { for (const v of o) collectRefs(v, out, seen); return; }
  if (o === null || typeof o !== 'object') return;
  const n = o as Record<string, unknown>;
  const ref = n['ref'];
  if (typeof ref === 'string' && ref.startsWith('n_') && !seen.has(ref)) {
    seen.add(ref);
    const bits = [`[${String(n['role'] ?? '?')}]`];
    for (const k of ['name', 'text', 'content', 'value']) {
      const v = n[k];
      if (typeof v === 'string' && v !== '') bits.push(v.slice(0, 70));
    }
    out.push({ ref, text: bits.join(' ').slice(0, 200) });
  }
  for (const v of Object.values(n)) collectRefs(v, out, seen);
}
async function likelyTargets(response: Record<string, unknown>, task: string,
                             hist: string[]): Promise<{ ref: string; score: number }[] | null> {
  if (NODE_RANKER === '') return null;
  const cands: { ref: string; text: string }[] = [];
  collectRefs(response, cands, new Set());
  if (cands.length < 4) return null;
  try {
    const r = await fetch(`${NODE_RANKER}/rank`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ query: `TASK: ${task} | SOFAR: ${hist.slice(-6).join(' ')}`,
                             candidates: cands }),
      signal: AbortSignal.timeout(4000),
    });
    if (!r.ok) throw new Error('ranker HTTP ' + r.status);
    const j = await r.json() as { ranked?: { ref: string; score: number }[] };
    if (!Array.isArray(j.ranked)) throw new Error('ranker returned no ranking');
    return j.ranked.slice(0, 8);
  } catch (e) {
    // FAIL CLOSED, by owner decision. Failing open is worse than useless for a
    // measurement: a dead sidecar yields episodes that silently ran WITHOUT the
    // ranker while the arm still reports a number, so the arm measures nothing
    // and looks fine. If the ranker is switched on it is present every episode
    // or the episode is void.
    throw new Error('node ranker unavailable (' + NODE_RANKER + '): ' + String((e as Error).message).slice(0, 120));
  }
}
const rankerHistory: string[] = [];

async function dispatchToolCall(
  session: WirSession, call: ToolCall, metrics: Metrics, traj: WriteStream,
  request: AgentRequest,
  confront: { finish: boolean; giveUp: boolean; giveUpSecond: boolean; verdictBounced: boolean },
): Promise<ToolOutcome> {
  const name = call.function.name;
  let args: Record<string, unknown>;
  try {
    args = JSON.parse(call.function.arguments || '{}') as Record<string, unknown>;
  } catch {
    return rejection(traj, call, 'tool arguments were not valid JSON');
  }
  if (name === 'give_up') {
    // Agent-local: never reaches session.dispatch, so the finish gate keeps its
    // full authority (ADR-003 untouched) and the episode is still scored.
    const reason = typeof args['reason'] === 'string' ? args['reason'] : '(no reason given)';
    // A SECOND BOUNCE WHEN THE AGENT IS DEMONSTRABLY HOLDING SOMETHING. The one
    // flag was the entire predicate, so an episode with a verified act in the
    // ledger quit exactly as easily as one with nothing. gateEligibleActs() was
    // already being fetched here purely to be printed; now it is also read.
    // Still not a gate — give_up stays reachable, and this cannot fire inside
    // the forced-finish window because the caller has cleared the flag by then.
    const holdsProof = session.gateEligibleActs().length > 0;
    if (confront.giveUp || (holdsProof && !confront.giveUpSecond)) {
      if (!confront.giveUp) confront.giveUpSecond = true;
      // Once per episode, and never when the model has no turn left — the caller
      // clears this on the same condition it clears the finish flag. A distinct
      // record kind because a bounced give_up is not an abstention: tools that
      // read `give_up` to mean "this episode ended in abstention"
      // (debug/mutate_ledger.mjs) must keep reading exactly that.
      confront.giveUp = false;
      record(traj, {
        kind: 'give_up_confrontation', reason, mode: session.expectedAction,
        ledger: session.gateEligibleActs(),
        unread: session.unconsumedContinuations(), finds: findPopulations,
      });
      return { content: giveUpConfrontation(session), finishAccepted: false,
        answer: '', finishStatus: null };
    }
    record(traj, { kind: 'give_up', reason });
    return { content: JSON.stringify({ gaveUp: true, reason }), finishAccepted: false,
      answer: '', finishStatus: null, gaveUp: { reason } };
  }
  if (name !== 'find' && name !== 'read' && name !== 'act' && name !== 'navigate' && name !== 'finish') {
    return rejection(traj, call, `unknown tool ${name}; the tools are find, read, act, navigate, finish, give_up`);
  }
  if (metrics.wirCalls >= request.budgets.maxWirCalls && name !== 'finish') {
    return rejection(traj, call, 'WIR call budget exhausted; only finish is allowed now');
  }
  // Before the confrontations, so a bounced finish still counts as attempted: the
  // budget checkpoint is for a model that has not reached for the end at all.
  if (name === 'finish') finishAttempted = true;
  // VALIDATE BEFORE COMPOSING THE CONFRONTATION. A finish sent with the wrong key
  // — `refs` where the schema says `evidenceRefs` — had the unknown key dropped,
  // and the confrontation then told the model "0 evidence refs cited" when four
  // had been supplied. That text is materially false, it costs a model call, the
  // verdict is approved on it, and only THEN does the runtime reject the call for
  // the malformed argument. Measured on shopping_admin 269.
  //
  // The predicate is deliberately CONSERVATIVE, and the direction matters: if it
  // called something malformed that the runtime would have accepted, that finish
  // would skip its one confrontation and ship unreviewed — a silent weakening far
  // worse than the bug being fixed. So it fires only on what the runtime
  // certainly rejects: evidenceRefs absent or not an array of strings. Anything
  // else still gets its confrontation exactly as before.
  const finishArgsMalformed = (): boolean => {
    if (name !== 'finish') return false;
    let parsed: unknown;
    try { parsed = JSON.parse(call.function.arguments || '{}'); } catch { return true; }
    if (typeof parsed !== 'object' || parsed === null) return true;
    const refs = (parsed as Record<string, unknown>)['evidenceRefs'];
    return !Array.isArray(refs) || refs.some(r => typeof r !== 'string');
  };
  // Fall through to session.dispatch, which produces the real, accurate rejection
  // naming the actual problem. `confront.finish` is deliberately NOT cleared, so
  // the first WELL-FORMED finish still gets its one confrontation.
  if (name === 'finish' && confront.finish && !finishArgsMalformed()) {
    // Once per episode, and never inside the forced-finish window — see the
    // caller, which clears this when the model has no turn left to spend.
    // Bouncing a model that cannot call again would destroy a finish it had
    // earned.
    if (session.expectedAction === 'MUTATE') {
      confront.finish = false;
      record(traj, {
        kind: 'evidence_confrontation', mode: 'MUTATE',
        arguments: call.function.arguments, ledger: session.gateEligibleActs(),
      });
      return { content: mutateConfrontation(session, call.function.arguments), finishAccepted: false,
        answer: '', finishStatus: null };
    }
    // RETRIEVE/NAVIGATE: the evaluator, on EVERY first finish — the loss class
    // it exists for has nothing partial about it, so a trigger conditioned on
    // partial delivery cannot see it. Two exceptions, both narrow:
    //   - an earned zero: status not_found_error with every cited population
    //     fully delivered is the model doing exactly what the system prompt
    //     teaches, and bouncing it would punish the honest path (unknowable
    //     facts do NOT qualify — null is not "fully delivered");
    //   - no room: the evaluator costs one model call, and it must never eat
    //     into the reserved forced-finish window. The caller already cleared
    //     the flag inside the window; this guards the approach to it.
    const refs = evidenceRefsOf(call.function.arguments);
    const facts = session.scopeFacts(refs);
    const status = typeof args['status'] === 'string' ? args['status'] : 'success';
    // ...but only when the model actually gave an account of what it found. A
    // finish whose answer is empty or is just the status token back again is not
    // the honest path being punished, it is no statement at all — and it is the
    // exact shape of shopping-329, where a total over an empty set was encoded as
    // absence twice running while EVALUATOR_SYSTEM names that very conflation as
    // a thing to refute. Measured: the guard was skipping the one finish the
    // critic existed to catch.
    const answerText = (typeof args['answer'] === 'string' ? args['answer'] : '').trim();
    const accounted = answerText !== '' && answerText.toLowerCase() !== status.toLowerCase();
    const earnedZero = status === 'not_found_error' && accounted
      && facts !== null && facts.every(f => f.delivered >= f.items);
    const room = metrics.modelCalls + 3 <= request.budgets.maxModelCalls;
    if (!earnedZero && room) {
      confront.finish = false;
      const answer = typeof args['answer'] === 'string' ? args['answer'] : '';
      let verdict: { approve: boolean; objection: string } | null = null;
      let evalResult: ChatResult | null = null;
      let evalError: string | null = null;
      try {
        const evalMessages: ChatMessage[] = [
          { role: 'system', content: EVALUATOR_SYSTEM },
          { role: 'user', content: evaluatorBrief(request.task, session,
            { answer, status, refCount: refs.length }, facts) },
        ];
        recordPayload('evaluator', metrics.modelCalls + 1, evalMessages, true);
        evalResult = await chatComplete(evalMessages, [VERDICT_TOOL],
          { type: 'function', function: { name: 'verdict' } });
        countModelCall(metrics, evalResult);
        trace?.generation({ model: modelName, input: evalResult.requestBody,
          output: evalResult.message, ms: evalResult.ms,
          usage: { input: evalResult.inputTokens, output: evalResult.outputTokens } });
        verdict = evaluatorVerdict(evalResult.message);
      } catch (error) {
        // Fail open: a dead or misbehaving evaluator must never cost the
        // episode its finish. The failure is on the record, never swallowed.
        evalError = describe(error);
      }
      record(traj, {
        kind: 'finish_evaluation',
        // The call number the evaluator spent, so the trajectory's numbering
        // has no silent gap (the contract record carries its own the same way).
        call: metrics.modelCalls,
        verdict: verdict === null ? (evalError !== null ? 'error' : 'unparseable')
          : verdict.approve ? 'approved' : 'rejected',
        ...(verdict !== null && !verdict.approve ? { objection: verdict.objection } : {}),
        ...(evalError !== null ? { error: evalError } : {}),
        content: evalResult?.message.content ?? null,
        arguments: call.function.arguments,
        facts, unread: session.unconsumedContinuations(),
        ms: evalResult?.ms ?? null,
        inputTokens: evalResult?.inputTokens ?? null,
        outputTokens: evalResult?.outputTokens ?? null,
      });
      if (verdict !== null && !verdict.approve) {
        return { content: objectionBounce(verdict.objection), finishAccepted: false,
          answer: '', finishStatus: null };
      }
      // AN UNPARSEABLE VERDICT IS NOT AN APPROVAL. It used to be: `verdict ===
      // null` fell through here and the finish proceeded, so a guard that ran and
      // produced garbage counted as a guard that passed. Measured on
      // human-57/task-57/attempt-12 — verdict "unparseable", answer 6 of 7
      // restaurants, accepted. The driver had guessed the tool's argument name
      // because the human seam was shown tool NAMES without schemas; that half is
      // fixed in wir-cli, and this is the other half.
      //
      // Every guard in the surveyed prior art fails CLOSED (NeMo Guardrails,
      // VLAA-GUI, LangGraph interrupts). Bounded to ONCE per episode, because a
      // model that cannot call the tool at all would otherwise be bounced on
      // every finish until its budget ran out — the exact "a guard must never
      // brick an episode" concern that motivated failing open. First unparseable
      // verdict costs one retryable finish; any later one fails open and is
      // recorded as before.
      //
      // A TRANSPORT failure (evalError) still fails open unconditionally: a dead
      // evaluator says nothing about the answer, and that distinction is the
      // whole point — this branch fires only when the evaluator REPLIED.
      if (verdict === null && evalError === null && !confront.verdictBounced) {
        confront.verdictBounced = true;
        record(traj, { kind: 'verdict_unparseable_bounce', call: metrics.modelCalls });
        return { content: VERDICT_UNPARSEABLE, finishAccepted: false,
          answer: '', finishStatus: null };
      }
      // Approved, failed open on transport, or already bounced once.
    } else {
      confront.finish = false;
    }
  }
  // Flat schema: verb + the model's flat arguments. The runtime validates for real.
  const verbRequest = { verb: name, ...args } as VerbRequest;
  const response = await wirDispatch(session, verbRequest, metrics, traj);
  // The offer rides INSIDE this tool result, on the same channel as the
  // repeat-rejection escalation and for the same reason (a user-role message
  // here would split a batch's tool block). Recorded with its fate: the next
  // act either carried the offered key or did not, and after two ignored
  // offers of a pattern the model is not shown a third.
  const { offer, outcome: offerOutcome } = offerAfter(offerState, name, args, response);
  if (offerOutcome !== null) record(traj, { kind: 'next_call_offer_outcome', ...offerOutcome });
  if (offer !== null) {
    record(traj, { kind: 'next_call_offer', pattern: offer.pattern, shape: offer.shape,
      ref: offer.ref, chars: offer.line.length, line: offer.line });
  }
  if (name === 'act' && typeof response['actRef'] === 'string') {
    actRecords.set(response['actRef'], { args, response });
    const action = args['action'];
    if ((action === 'fill' || action === 'type' || action === 'select') && typeof args['value'] === 'string'
        && args['value'].trim() !== '') {
      filledValues.add(args['value'].trim());
    }
  }
  // A site skill takes the attachment slot when it is due — at most once per
  // origin per episode, so in practice it never competes with the screenshot arm.
  // It must NOT short-circuit the return: an early exit here would hardcode
  // finishAccepted:false and swallow an accepted finish on the one call where
  // both could land.
  const skill = siteSkill(String((response as Record<string, unknown>)['url'] ?? session.host.page.url()));
  if (skill !== null) {
    record(traj, { kind: 'site_skill', origin: session.host.page.url(), bytes: skill.length });
  }
  // FENCED AND ATTRIBUTED, because a bare user turn arriving right after a tool
  // result reads as more page content — the model has no way to tell notes from
  // observations. The system prompt introduces this block by name, so the tag is
  // a concept the model already has when the first one appears, and `origin`
  // scopes it: on a two-site task both blocks stay in context forever, and the
  // one that applies is the one matching the response's `url`, not the most
  // recent. Measured across three two-site episodes with both blocks live: no
  // cross-site vocabulary — but every one of them travelled shop -> forum and
  // never back, so the interleaved case is untested rather than proven safe.
  const origin = (() => {
    try { return new URL(session.host.page.url()).origin; } catch { return 'unknown'; }
  })();
  const skillMessage: ChatMessage | null = skill === null ? null : { role: 'user', content:
    `<site_knowledge origin="${origin}">\n${skill}\n</site_knowledge>` };
  const attachment = skillMessage ?? (name === 'read' && READ_SCREENSHOT
    ? await readScreenshotMessage(session, response, (readCount += 1))
    : null);
  rankerHistory.push(name);
  const hints = await likelyTargets(response, String(
    (request.task as Record<string, unknown> | undefined)?.['instruction'] ?? ''), rankerHistory);
  const withHints = hints && hints.length > 0
    ? { ...response, likelyTargets: { note: 'advisory local ranking; the full list above is unchanged', refs: hints } }
    : response;
  // THE TASK, RESTATED WHERE ATTENTION IS. Delivered once in message 2 and then
  // buried under hundreds of KB of observations — and the largest failure family
  // in the corpus is "did the right kind of work, committed to the wrong thing":
  // of 387 recorded failures, 223 carry a `missing_navigation_event` assertion and
  // 170 carry it as the ONLY one — and three map losses were choosing the wrong
  // instance of an ambiguous name (the first hotel of ten, the wrong Hyatt twice,
  // a Starbucks an hour and a half away).
  //
  // Corrected 2026-08-20. This said "117 of 120 REQUEST_MISMATCH failures". Both
  // halves were wrong: the figures do not reproduce, and REQUEST_MISMATCH is not a
  // code-defined outcome anywhere — it existed only in this comment and in the
  // copy of it in test/task-reminder.test.ts. Re-derived over 898 scored episodes
  // across benchmark-results/ and every /home/opc/wir-runs-archive root.
  //
  // On EVERY observation, not just a committing act: the model composes an act
  // before seeing any response, so a reminder attached to the act result arrives
  // after the commit and on a MUTATE may be unrecoverable. Here it sits at the end
  // of the turn where the choice is actually made.
  //
  // The task's own words, verbatim — no extraction, no token matching, no
  // "identifying constraint" inference. That would be a comparator on the critical
  // path, which is banned outright (CLAUDE.md, "No semantic matcher"). No task data
  // reaches core/ either: this is the agent layer, where the instruction already
  // lives.
  //
  // PRE-REGISTERED, AND PREDICTED TO FAIL. debug/LEARNINGS.md records that advisory
  // signals do not move this model — resolvedDifferently was built for this exact
  // failure, was delivered in all five losses, and was read past every time, once
  // immediately after the driver had been warned about that precise mistake. If a
  // controlled arm shows no conversion, DELETE this rather than leave a third
  // unread signal in the payload.
  const instruction = String((request.task as Record<string, unknown> | undefined)?.['instruction'] ?? '');
  const payload = instruction === ''
    ? withHints
    : { ...withHints, taskReminder: { task: instruction } };
  return {
    ...(attachment !== null ? { attachment } : {}),
    content: `${JSON.stringify(payload)}${offer !== null ? `\n${offer.line}` : ''}`,
    finishAccepted: name === 'finish' && response['accepted'] === true,
    answer: name === 'finish' && typeof args['answer'] === 'string' ? args['answer'] : '',
    finishStatus: name === 'finish' && typeof response['status'] === 'string'
      ? response['status'] : null,
  };
}

// Where the declared answer shape goes, and why here rather than in the finish
// tool description or a confrontation.
//
// The task message is the RUNNER's statement of the episode — the instruction and
// where the browser already is — and the shape is one more thing the runner knows
// and the model does not. Three properties the alternatives do not have. It sits
// in the cached prefix, so it costs nothing after call 1 and invalidates nothing
// (the transcript is append-only for exactly that reason). It arrives before the
// first read, so it shapes WHAT the model goes and collects — a schema of
// {book, author} pairs is an instruction to record authors while reading — where
// the finish tool description would reach it only on the last call, after the
// evidence was already gathered in the wrong shape, and the pre-finish
// confrontation fires once and only on a first finish attempt. And no task data
// enters core/: the tool schemas stay task-independent, which is what makes them
// one source of truth for every consumer (core/toolschemas.ts).
//
// The last sentence is load-bearing. The schema describes form; a model that
// reads it as a hint about content would be reading the evaluator's mind, and the
// runtime never compares an answer to anything (ADR-003).
function taskMessage(task: AgentRequest['task']): string {
  let base = `Task: ${task.instruction}\nThe browser is already on the task's start page.`;
  // A task may span two sites, and nothing on one links to the other. Seeding
  // the second URL into the runtime's observed set makes `navigate` ACCEPT it;
  // it does not make the model AWARE of it. On the first multi-site run that
  // distinction cost the whole episode: the agent hunted for a forum on the
  // storefront, tried `localhost:7770/forum`, was correctly refused, searched
  // the shop for the word "forum", and gave up — with the forum sitting at a
  // port it was never told about.
  const others = (task.startUrls ?? []).filter(u => u !== task.startUrl);
  if (others.length > 0) {
    base += `\nThis task spans more than one site. The others are reachable with `
      + `navigate: ${others.join(', ')}`;
  }
  const schema = task.resultsSchema;
  if (schema === undefined || schema === null) return base;
  // The encoding sentence is load-bearing. `answer` is typed `string` on the
  // finish tool, and this paragraph announces a schema that is often an ARRAY —
  // so "must be JSON that validates against it" reads, correctly and fatally, as
  // "pass an array", and the call is rejected `invalid_args`. Two schemas are
  // true at once and nothing here used to reconcile them. Measured on the
  // shopping_admin population: 82 of 182 tasks declare an array answer, and the
  // first hand-driven episode to meet one spent three finish calls discovering
  // the encoding by trial — refs piled on, `status` reached for — before landing
  // on the string form. The rejection says what is wrong; it just says it after
  // the budget is spent, and a ~60-call agent pays the same toll.
  return `${base}\n\nAnswer shape — the JSON Schema this task declares for its answer: ${JSON.stringify(schema)}\n` +
    'Your finish answer must be JSON that validates against it: exactly these keys and ' +
    'types, nothing extra, no prose. It describes the FORM of the answer only — which ' +
    'values belong in it is still yours to find on the page.\n' +
    'The finish tool\'s `answer` parameter is a STRING, so send that JSON encoded as a ' +
    'string — for an array schema that is answer: "[\\"first\\", \\"second\\"]", not ' +
    'answer: ["first", "second"], which is rejected. `evidenceRefs` stays a real array ' +
    'and is never encoded.';
}

// Every completion is billed the same way, wherever it is made — the loop, the
// contract, the finish evaluator. One accounting site, so an extra call class
// can never silently fall out of modelCalls (the number a budget is declared in).
function countModelCall(metrics: Metrics, result: ChatResult): void {
  metrics.modelCalls += 1;
  metrics.inputTokens += result.inputTokens;
  metrics.outputTokens += result.outputTokens;
  metrics.providerLatencyMs += result.ms;
  if (result.cachedTokens !== null) {
    metrics.cachedInputTokens = (metrics.cachedInputTokens ?? 0) + result.cachedTokens;
  }
  if (result.reasoningTokens !== null) {
    metrics.reasoningTokens = (metrics.reasoningTokens ?? 0) + result.reasoningTokens;
  }
}

async function drive(
  session: WirSession, request: AgentRequest, metrics: Metrics, traj: WriteStream,
): Promise<{ status: 'success' | 'budget_exhausted' | 'agent_abstained';
             answer: string; finishStatus: string | null }> {
  const { budgets, task } = request;
  const started = Date.now();
  const deadline = started + budgets.maxWallTimeMs;
  const softDeadline = started + Math.floor(budgets.maxWallTimeMs * 0.9);
  const messages: ChatMessage[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: taskMessage(task) },
  ];
  let forcedFinal = false;
  let promptLoggedCount = 0;
  let batchNudges = 0;
  // Two thirds of the DECLARED model-call budget, counted in the same unit the
  // budget is declared in — every completion, the contract turn and the finish
  // evaluator included (countModelCall). At the harness default of 60 that is
  // call 40, with the forced window still 18 calls away.
  const checkpointAt = Math.floor(budgets.maxModelCalls * 2 / 3);
  // One pre-finish confrontation per episode, whatever the mode: MUTATE gets the
  // proof ledger, RETRIEVE/NAVIGATE the coverage accounting. give_up carries its
  // own, on the same terms — the two are separate decisions and one bounce each.
  const confront = { finish: true, giveUp: true, giveUpSecond: false, verdictBounced: false };
  // The transcript only grows, so the last call's input size is a lower bound on
  // the next one's. Checking it BEFORE the call turns "the model exhausted its
  // own context" into this agent's scored budget_exhausted, instead of a
  // provider 4xx -> provider_error -> EXCLUDED, which shrinks the denominator
  // and inflates the pass rate with the agent's own failure. No provider-error
  // string matching: the budget is a number the runner declares.
  const maxInputTokens = budgets.maxInputTokensPerCall ?? null;
  let lastInputTokens = 0;
  // A provider that omits `usage` reports 0 input tokens, which would make the
  // budget silently unenforceable — a safety mechanism that looks armed and is
  // not. Say so once, loudly, rather than letting a declared budget quietly mean
  // nothing (unknown is not zero).
  let warnedNoUsage = false;
  // THE COMPLETION CONTRACT: one text exchange before the first tool turn.
  // Never on MUTATE (the proof ledger is that mode's confrontation), and never
  // when the budget is so small the forced window would swallow it — the
  // contract costs one call, and at maxModelCalls <= 3 the first tool turn
  // after it would already be forced (modelCalls 1 + 2 >= max).
  if (task.expectedAction !== 'MUTATE' && budgets.maxModelCalls > 3) {
    messages.push({ role: 'user', content: COMPLETION_CONTRACT });
    recordPayload('contract', metrics.modelCalls + 1, messages);
    const result = await chatComplete(messages, TOOLS);
    countModelCall(metrics, result);
    trace?.generation({ model: modelName, input: result.requestBody,
      output: result.message, ms: result.ms,
      usage: { input: result.inputTokens, output: result.outputTokens } });
    lastInputTokens = result.inputTokens;
    messages.push(result.message);
    const text = typeof result.message.content === 'string' ? result.message.content : '';
    contractText = text.trim() === '' ? null : text;
    // The one turn where a bare text reply IS the request, so no "Respond with
    // a tool call" nudge. A tool call issued here is refused, not run — and
    // every tool_call_id still gets an answer, because an unanswered one is a
    // protocol error on the next request.
    const stray = result.message.tool_calls ?? [];
    for (const c of stray) {
      messages.push({ role: 'tool', tool_call_id: c.id, content: JSON.stringify({
        rejected: { kind: 'invalid_args',
          reason: 'this turn asks for your completion criteria as plain text; no tool was run' } }) });
    }
    record(traj, { kind: 'completion_contract', call: metrics.modelCalls,
      ms: result.ms, inputTokens: result.inputTokens, outputTokens: result.outputTokens,
      content: result.message.content,
      refusedToolCalls: stray.map(c => c.function.name) });
  }
  while (metrics.modelCalls < budgets.maxModelCalls && Date.now() < deadline) {
    const overContextBudget = maxInputTokens !== null && lastInputTokens >= maxInputTokens;
    // The forced-finish window is the last TWO calls, not the last one. The
    // finish gate teaches through a repair string, and a forced finish it
    // rejects used to arrive with the loop condition already false: the model
    // read "cite an act whose evidence proves a server-side change" and never
    // got to act on it (failed10-current-1/task-442/attempt-1 and .../task-659,
    // failed10-proj/task-442/attempt-1 — finish rejected on call 60 of 60, then
    // budget_exhausted). Every other mustFinish trigger already leaves calls in
    // hand and so already allows a repair; this one did not, so it is the one
    // that changes shape.
    //
    // Reserving a call rather than granting one past the cap: maxModelCalls is
    // the RUNNER's declaration, and an agent that spends 61 of a declared 60 has
    // taken authority the harness gives it (docs/benchmark.md). It costs one
    // working call in episodes that reach the cap.
    const mustFinish =
      metrics.modelCalls + 2 >= budgets.maxModelCalls ||
      metrics.wirCalls >= budgets.maxWirCalls ||
      overContextBudget ||
      Date.now() > softDeadline;
    // A bounce costs the model a turn, so only spend one when a re-submit is
    // certainly still possible. The forced window now subsumes the old separate
    // two-spare-calls guard (modelCalls + 2 > maxModelCalls implies the mustFinish
    // clause above), so one condition covers both terminal tools: bouncing a
    // model that cannot call again destroys a finish the gate would have
    // accepted, and turns an honest abstention into budget_exhausted.
    if (mustFinish) {
      confront.finish = false;
      confront.giveUp = false;
      confront.giveUpSecond = true;
    }
    if (mustFinish && !forcedFinal) {
      forcedFinal = true;
      messages.push({ role: 'user', content: FORCE_FINISH });
      if (overContextBudget) {
        log(`context budget reached (${lastInputTokens} >= ${maxInputTokens} input tokens); forcing finish`);
      }
    } else if (overContextBudget) {
      // The forced final call already fired and the transcript is still over
      // budget: stop. The exemption is exactly one call — without it the agent
      // could never finish at all, and with more than one it would keep making
      // the calls the budget exists to prevent.
      log(`context budget still exceeded after the forced finish; stopping as budget_exhausted`);
      break;
    }
    // THE BUDGET CHECKPOINT, once, and structurally unable to collide with
    // FORCE_FINISH: it is guarded on !mustFinish, which is the exact condition
    // that fires the forced window, and it sits after that branch so even a
    // future edit to the guard cannot get it in first. Every other skip is a
    // precondition of the thing being injected — no contract, no criteria to
    // restate; a finish already attempted, and the end-of-episode machinery owns
    // the model instead.
    if (BUDGET_CHECKPOINT && !budgetCheckpointFired && !mustFinish && !forcedFinal
        && !finishAttempted && contractText !== null
        && metrics.modelCalls >= checkpointAt) {
      budgetCheckpointFired = true;
      const content = budgetCheckpointMessage(
        contractText, metrics.modelCalls, budgets.maxModelCalls);
      messages.push({ role: 'user', content });
      record(traj, {
        kind: 'budget_checkpoint', call: metrics.modelCalls,
        modelCalls: metrics.modelCalls, maxModelCalls: budgets.maxModelCalls,
        checkpointAt, wirCalls: metrics.wirCalls, chars: content.length,
      });
      log(`budget checkpoint at ${metrics.modelCalls}/${budgets.maxModelCalls} model calls`);
    }
    if (LOG_PROMPT) {
      // Images stripped: a screenshot is ~250KB of base64 per call, and the point
      // of this dump is to read what the model received. The bytes are on disk
      // under the attempt's screenshots/ directory.
      const fresh = messages.slice(promptLoggedCount)
        .map(m => ({ ...m, content: stripImageParts(m.content) }));
      log(`[prompt] call ${metrics.modelCalls + 1} (+${fresh.length} messages): ${JSON.stringify(fresh)}`);
      promptLoggedCount = messages.length;
    }
    recordPayload('loop', metrics.modelCalls + 1, messages);
    const result = await chatComplete(messages, TOOLS);
    countModelCall(metrics, result);
    trace?.generation({ model: modelName, input: result.requestBody,
      output: result.message, ms: result.ms,
      usage: { input: result.inputTokens, output: result.outputTokens } });
    lastInputTokens = result.inputTokens;
    if (maxInputTokens !== null && result.inputTokens === 0 && !warnedNoUsage) {
      warnedNoUsage = true;
      log('WARNING: maxInputTokensPerCall is declared but the provider reported 0 ' +
          'input tokens — it is omitting usage, so the context budget CANNOT fire');
    }
    messages.push(result.message);
    const toolCalls = result.message.tool_calls ?? [];
    metrics.toolCallsIssued += toolCalls.length;
    if (toolCalls.length > 1) metrics.batchedTurns += 1;
    record(traj, {
      kind: 'model', call: metrics.modelCalls, ms: result.ms,
      inputTokens: result.inputTokens, outputTokens: result.outputTokens,
      cachedTokens: result.cachedTokens,
      // 94.8% of billed output tokens appear nowhere in the recorded artifacts
      // (692,520 of 730,704 over 5,211 calls). These two fields are what close
      // that. reasoningTokens is null, not 0, when the provider omits it, so a
      // non-reasoning model reads differently from an unread field. The text is
      // bounded because it is the largest thing on the record and the trajectory
      // is read by hand — the exact residual is stated, never a bare cut.
      reasoningTokens: result.reasoningTokens,
      reasoning: result.reasoningText === null ? null
        : result.reasoningText.length <= REASONING_RECORD_CHARS
          ? result.reasoningText
          : `${result.reasoningText.slice(0, REASONING_RECORD_CHARS)}`
            + `\u2026[+${result.reasoningText.length - REASONING_RECORD_CHARS} chars]`,
      content: result.message.content,
      toolCalls: toolCalls.map((t) => t.function.name),
    });
    if (toolCalls.length === 0) {
      messages.push({ role: 'user', content: 'Respond with a tool call. Use finish to end the episode.' });
      continue;
    }
    // A BATCH STOPS AT THE FIRST CALL THAT MOVES THE PAGE.
    //
    // Within one batch the model cannot have used call N's result to compose call
    // N+1 — it emitted them together, before seeing any of them. So no call in a
    // batch depends on another's output, and the only hazard is a call that CHANGES
    // THE PAGE: every later call in the batch was composed against a document that
    // no longer exists, and its refs were minted under an epoch that is now dead.
    //
    // Running them anyway is how batching would turn into a rejection generator —
    // 206 stale_ref/unknown_ref rejections already land on the call immediately
    // after an act, and that is the population this pushes on. So the remainder is
    // skipped and SAID to be skipped: every tool_call_id still gets an answer,
    // because an unanswered one is a protocol error on the next request.
    let pageMoved: string | null = null;
    for (const call of toolCalls) {
      if (pageMoved !== null) {
        messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify({
          rejected: { kind: 'invalid_args',
            reason: `not run: an earlier call in this batch (${pageMoved}) moved the page, `
              + 'so the refs this call was composed against no longer exist',
            repair: '{"verb":"read"} — see where you are now, then re-issue this call' } }) });
        continue;
      }
      const outcome = await executeToolCall(session, call, metrics, traj, request, confront);
      messages.push({ role: 'tool', tool_call_id: call.id, content: outcome.content });
      const name = call.function.name;
      // "Moved the page" is judged by WHAT HAPPENED, not by the verb's name.
      //
      // Stopping on every `act` made filling a six-field form cost six model
      // turns — at ~5s of provider latency each, the form was 30 seconds of
      // waiting before the submit. Measured across 23 episodes: 463 model turns
      // carried 577 tool calls, and 401 of those turns carried exactly one. The
      // batching rule was asking for concurrency the loop then refused.
      //
      // A `fill` that sets its own field's value invalidates nothing: the refs
      // in the rest of the batch still name the same nodes. Only an act whose
      // effect reaches BEYOND its target can strand them — a navigation, a
      // client-side re-render, a dialog. Those still stop the batch.
      //
      // The safety net underneath is unchanged and is what makes this sound:
      // every act revalidates its ref against the browser's own computation at
      // dispatch, so a ref that did go stale is REFUSED with `stale_ref`, not
      // acted on blindly. The worst case is a typed rejection the model is told
      // about; the old worst case was a wasted turn on every field.
      if (name === 'navigate') pageMoved = name;
      if (name === 'act' && !effectStayedLocal(outcome.content)) pageMoved = name;
      // Append-only, always. The prefix cache serves ~95% of input tokens at 1/50th
      // price; dropping or rewriting an earlier image would make every token after
      // it a miss forever. Staleness is handled by the label and the prompt rule.
      if (outcome.attachment !== undefined) messages.push(outcome.attachment);
      if (outcome.finishAccepted) {
        return { status: 'success', answer: outcome.answer, finishStatus: outcome.finishStatus };
      }
      if (outcome.gaveUp) {
        log(`agent gave up: ${outcome.gaveUp.reason}`);
        return { status: 'agent_abstained', answer: '', finishStatus: null };
      }
    }
    // The turn is over and the page did not move, so every one of these calls
    // could have travelled with the next turn's. This is the denominator a
    // batching fix has to move, recorded per episode rather than re-derived.
    if (pageMoved === null && toolCalls.length > 0) {
      metrics.unmovedTurns += 1;
      // TURN-LOCAL FEEDBACK, on exactly the turns where it applies. A static
      // prompt rule asking for batches has sat in the first three of 32 rules
      // for weeks and moved the rate to ~4-5% of turns against a 58.3%
      // opportunity, and the rate is FLAT across the episode (3.1% at turns 1-2,
      // 5.3% at 21+), so "the instruction got buried" is refuted — it is simply
      // not salient at the moment of composing a turn. This says it at that
      // moment, once per turn, only when the turn really was mergeable.
      //
      // Appended INSIDE the last tool result, never as a user message: a
      // user-role message here would split the tool block for the calls still
      // being answered. Same channel as the repeat-rejection escalation.
      const last = messages[messages.length - 1];
      if (toolCalls.length === 1 && batchNudges < BATCH_NUDGE_LIMIT
          && last !== undefined && last.role === 'tool'
          && typeof last.content === 'string') {
        // A lone field write is mergeable with the form's OTHER fields, and NOT
        // with the click that submits — the prompt puts that in the next turn on
        // purpose. Saying "your next one" after a fill therefore asked for the one
        // merge the rules forbid, so a driver obeying the prompt got nagged and a
        // driver obeying the nag broke the rule. Reported by a driver on task 12.
        const lone = toolCalls[0];
        let loneAction = '';
        if (lone !== undefined && lone.function.name === 'act') {
          try {
            const parsed: unknown = JSON.parse(lone.function.arguments);
            if (typeof parsed === 'object' && parsed !== null) {
              loneAction = String((parsed as { action?: unknown }).action ?? '');
            }
          } catch { loneAction = ''; }
        }
        const wroteAField = loneAction === 'fill' || loneAction === 'select'
          || loneAction === 'type' || loneAction === 'upload';
        // An act that is NOT a field write may have revealed something — a
        // disclosure opening, a scroll bringing rows in — and the next call is
        // chosen from what appeared. Telling that turn it could have travelled
        // with the next one asks for a batch composed against nodes that did not
        // exist when it was written. Reported on gitlab 414: a click that took
        // the graph from 254 to 271 nodes, verified target_state_changed, was
        // nudged anyway, and the driver's note reads "taken literally the hint
        // invites batching a find across a disclosure that has to open first —
        // exactly the wrong lesson". target_state_changed and scrolled are in
        // LOCAL_EFFECT_EVIDENCE because they leave other refs VALID, which is the
        // right call for batching and the wrong premise for this nudge.
        const revealedSomething = lone !== undefined
          && lone.function.name === 'act' && !wroteAField;
        if (!revealedSomething) {
          batchNudges += 1;
          const note = wroteAField
            ? 'that turn wrote one field and moved nothing, so the form\'s OTHER '
              + 'fields could have travelled with it — fill, select, type and upload '
              + 'for every field of a form belong in ONE turn. The click that submits '
              + 'does NOT: it stays in the turn after, exactly as your rules say.'
            : 'that turn changed nothing on the page, so this call and your next one '
              + 'could have travelled together. Independent calls belong in ONE turn — '
              + 'every field of a form, or several finds you already know you need.';
          last.content = `${last.content}\n${JSON.stringify({ note })}`;
        }
      }
    }
  }
  return { status: 'budget_exhausted', answer: '', finishStatus: null };
}

export async function runEpisode(request: AgentRequest): Promise<EpisodeResult> {
  const metrics: Metrics = {
    modelCalls: 0, inputTokens: 0, outputTokens: 0, wirCalls: 0, browserDeliveries: 0,
    providerLatencyMs: 0, observationBytes: 0, cachedInputTokens: null, estimatedCost: null, reasoningTokens: null,
    readAfterRead: 0, readScreenshot: READ_SCREENSHOT, temperature, seed,
    toolCallsIssued: 0, batchedTurns: 0, unmovedTurns: 0, ...buildIdentity(),
    modelId: modelName,
  };
  // WirSession.start has no slowMo option, so the browser actually runs at slowMoMs 0.
  const effectiveBrowser = { headless: request.browser.headless, slowMoMs: 0 };
  let status: AgentStatus = 'agent_failed';
  let answer = '';
  let finishStatus: string | null = null;
  let traj: WriteStream | null = null;
  let session: WirSession | null = null;
  const telemetry = await createTelemetry().catch(() => noopTelemetry);
  // The episode runs INSIDE the trace scope: v5 propagates session/tags only to
  // spans created within it. `trace` is the module-level handle the seams use.
  return telemetry.run({
    runId: request.runId, taskId: request.task.taskId,
    instruction: request.task.instruction,
    expectedAction: request.task.expectedAction,
    benchmark: request.task.benchmark,
    model: modelName,
  }, async (episodeTrace) => {
  trace = telemetry.enabled ? episodeTrace : null;
  if (telemetry.enabled) log(`telemetry on (trace ${trace?.traceId ?? 'pending'})`);
  try {
    mkdirSync(dirname(request.artifacts.trajectoryPath), { recursive: true });
  initScreenshotsDir(request.artifacts.trajectoryPath);
    traj = createWriteStream(request.artifacts.trajectoryPath, { flags: 'w' });
    // An unhandled 'error' on a WriteStream throws asynchronously and would kill
    // the process outside any try — the trajectory is an artifact, never worth
    // the episode (review, cluster D).
    traj.on('error', (e) => log(`trajectory write failed (non-fatal): ${describe(e)}`));
    payloads = createWriteStream(
      `${dirname(request.artifacts.trajectoryPath)}/provider-payloads.jsonl`, { flags: 'w' });
    payloads.on('error', (e) => log(`payload log write failed (non-fatal): ${describe(e)}`));
    record(payloads, {
      kind: 'episode', runId: request.runId, taskId: request.task.taskId,
      // Identical on every main-loop call, so it is stated once here rather than
      // repeated per line.
      model: modelName, temperature, seed, toolChoice: 'auto',
      // The FULL definitions, not just the names. Tool schemas are part of what
      // the model receives — a description is prompt text sitting next to the
      // JSON it emits — and recording only names left the artifact unable to
      // answer "did this wording reach the model", which is the whole point of
      // the file. Identical on every call, so one copy in the header.
      parallelToolCalls: true, tools: TOOLS,
    });
    record(traj, {
      kind: 'episode', runId: request.runId,
      taskId: request.task.taskId, expectedAction: request.task.expectedAction,
      // Which arm produced this episode, in the episode's own artifact. Same rule
      // as metrics.readScreenshot and effectiveBrowser: a switch that changes what
      // a run measures must be readable off the run, not remembered.
      nudges: { repeatRejection: REPEAT_NUDGE, budgetCheckpoint: BUDGET_CHECKPOINT },
    });
    session = await startSession(request);
    const outcome = await drive(session, request, metrics, traj);
    status = outcome.status;
    answer = outcome.answer;
    finishStatus = outcome.finishStatus;
  } catch (error) {
    status = error instanceof ProviderError ? 'provider_error'
      : error instanceof BrowserError ? 'browser_error'
      : 'agent_failed';
    log(`episode error (${status}): ${describe(error)}`);
  } finally {
    // Close the trace before the browser: the root observation must end after its
    // children and before the process can exit. Two flushes, one per transport.
    if (trace !== null) {
      trace.end({ status, answer, modelCalls: metrics.modelCalls, wirCalls: metrics.wirCalls });
      const traceId = trace.traceId;
      trace = null;
      if (traceId !== null) {
        try {
          writeFileSync(`${dirname(request.artifacts.trajectoryPath)}/langfuse-trace.json`,
            JSON.stringify({ traceId, runId: request.runId }, null, 2));
        } catch { /* artifact is a convenience, never fatal */ }
      }
    }
    await telemetry.shutdown();
    if (session !== null) await session.close().catch((e) => log(`close failed: ${describe(e)}`));
    metrics.estimatedCost = estimateCost(metrics);
    if (traj !== null) {
      record(traj, { kind: 'end', status, ...metrics });
      await new Promise<void>((resolve) => traj?.end(resolve));
    }
    if (payloads !== null) {
      const stream = payloads;
      await new Promise<void>((resolve) => stream.end(resolve));
    }
    try {
      writeFileSync(request.artifacts.metricsPath, `${JSON.stringify(metrics, null, 2)}\n`);
    } catch (error) {
      log(`metrics write failed: ${describe(error)}`);
    }
  }
  return {
    status,
    finalResponse: evaluatorEnvelope(request.task.expectedAction, status === 'success', answer, finishStatus),
    metrics,
    effectiveBrowser,
  };
  });
}
