"""Deterministic trajectory recording and replay of an agent run.

A *trajectory* is the recorded story of one run: every model decision (the Pydantic AI response, serialized verbatim),
every Python cell the model wrote, the selectors that cell names, a bounded DOM snapshot digest taken before and after
the cell, and the tool result the REPL returned (redacted). Replay drives a fresh browser with the recorded decisions:
before each cell the live page is checked against the recorded digest, and any mismatch (a changed page, a selector that
no longer resolves, a corrupted or truncated trajectory, a different model family) ends replay and hands the run back to
the live model instead of executing a wrong action. From that point the run branches: new decisions are recorded into a
child trajectory, and ReplayResult.diff reports every divergence.

Design constraints honored here:
- recording is opt-in (Agent(trajectory_path=...)); without it the live loop is unchanged;
- storage is bounded (TRAJ_MAX_STEPS steps / TRAJ_MAX_BYTES per file, snapshots capped) and append-only JSONL;
- nothing serialized may contain a secret value: every line passes redact_forms() before it reaches disk, and image
  binaries are never stored;
- replay never rebuilds the whole page per step: one small page_info()/js probe per step, no full DOM serialization;
- a damaged file never kills a run: malformed lines are skipped and marked, an unreadable header degrades to live.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_core import to_jsonable_python

from browser_use.repl.session import redact_forms, secret_forms

if TYPE_CHECKING:
	from pydantic_ai.messages import ModelResponse

logger = logging.getLogger(__name__)
TRAJ_FORMAT = 'browser-use-trajectory/1'
TRAJ_MAX_STEPS = 500  # a run past this many recorded steps drops the oldest steps (the header and tail stay)
TRAJ_MAX_BYTES = 10 * 1024 * 1024  # and the whole file is bounded; the bound trumps the step count
SNAPSHOT_MAX_CHARS = 4_000  # a DOM snapshot summary, not the page: text beyond this is folded into the digest anyway
RESULT_MAX_CHARS = 12_000  # a tool result on disk: the same cut the worker gives the model
SELECTOR_LIMIT = 50  # selectors named in one cell, cap so a pathological cell cannot blow the bound alone
# What the snapshot probe returns. One round trip per cell boundary, page_info plus one js() read: never a full DOM dump.
_SNAPSHOT_JS = r"""
(function () {
  function sig(el) {
    if (!el) return null;
    var a = el.attributes, out = [];
    for (var i = 0; i < a.length && out.length < 8; i++) {
      var n = a[i].name;
      if (n === 'id' || n === 'name' || n === 'data-testid' || n.substr(0, 5) === 'aria-' || n === 'role'
          || n === 'type' || n === 'href' || n === 'placeholder') out.push(n + '=' + a[i].value);
    }
    return el.tagName.toLowerCase() + '[' + out.join(',') + ']';
  }
  var inputs = Array.prototype.slice.call(document.querySelectorAll('input,textarea,select,button,a[href]')).slice(0, 200)
    .map(sig);
  var body = document.body ? String(document.body.innerText || '').replace(/\s+/g, ' ').slice(0, 2000) : '';
  return {url: location.href, title: document.title, ready: document.readyState,
          element_count: document.querySelectorAll('*').length, inputs: inputs, text_head: body};
})()
"""
# Selectors the model's code hands to the helpers/raw CDP that address an element. Text arguments (type_text, goto) are
# not selectors: only strings that look like CSS/XPath or are keyword arguments named for an element locator.
_SELECTOR_CALL = re.compile(
	r'\b(?:querySelector(?:All)?|wait_for_element|fill_input|click|bu\.click|bu\.type|bu\.fill|bu\.select|find_element)'
	r'\s*\(\s*(["\'])(.*?)\1',
	re.S,
)
_KW_SELECTOR = re.compile(r'(?:selector|css|xpath)\s*=\s*(["\'])(.*?)\1', re.S)
_DOM_CALL = re.compile(r"\bDOM\.\w+\s*\([^)]*selector\s*=\s*(['\"])(.*?)\1", re.S)
# A CSS selector's structural/meta characters, or an XPath's prefix: plain words, URLs and bare text arguments are not
# treated as selectors.
_SELECTOR_SHAPE = re.compile(r'^(\.\.?|#|\*|/|[a-zA-Z][\w-]*\[|[^>]*[#.:>\[\]])')


def extract_selectors(code: str) -> list[str]:
	"""The element selectors a cell's code names (querySelector/wait_for_element/fill_input/bu.click/DOM.querySelector...).
	Deduplicated, order kept, bounded. These are what can go stale when the page changes, so replay probes each one."""
	out: list[str] = []
	for pat in (_SELECTOR_CALL, _KW_SELECTOR, _DOM_CALL):
		for m in pat.finditer(code):
			sel = m.group(2).strip()
			if sel and (sel.startswith(('/./', '//', 'xpath', '/')) or _SELECTOR_SHAPE.match(sel)):
				if sel not in out:
					out.append(sel)
			if len(out) >= SELECTOR_LIMIT:
				return out
	return out


def _digest(value: Any) -> str:
	"""Stable 16-byte blake2b digest of JSONable data (surrogate-safe): the fingerprint compared at replay time."""
	blob = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode('utf-8', 'replace')
	return hashlib.blake2b(blob, digest_size=16).hexdigest()


def clip_snapshot(value: Any) -> dict[str, Any]:
	"""Bound a snapshot's text fields; structural fields are already bounded by the probe itself."""
	if not isinstance(value, dict):
		return {}
	out: dict[str, Any] = {}
	for k, v in value.items():
		if isinstance(v, str) and len(v) > SNAPSHOT_MAX_CHARS:
			out[k] = v[:SNAPSHOT_MAX_CHARS]
		elif isinstance(v, list) and len(v) > 200:
			out[k] = v[:200]
		else:
			out[k] = v
	return out


def snapshot_digest(snapshot: dict[str, Any]) -> str:
	return _digest(snapshot)


def _model_family(model_name: str) -> str:
	"""The provider family a model name implies: the first segment of 'provider:model', so 'function:fn:' and
	'function:x' are one family and 'anthropic:claude-...' is another. Empty for an unnamed model (no constraint)."""
	if not isinstance(model_name, str) or ':' not in model_name:
		return model_name or ''
	return model_name.split(':', 1)[0]


@dataclass
class TrajectoryConfig:
	"""Public options for Agent(trajectory=...).

	path: where the JSONL trajectory is written (record) or read (replay). mode: 'record' (default) writes a new
	trajectory for a live run; 'replay' replays an existing one and branches to live decisions on the first mismatch;
	'replay-branch' additionally writes the branched live decisions into `branch_path` (default: <path>.branch.jsonl).
	seed: deterministic seed applied to the in-process random generators at run start (model sampling itself is
	provider-side; replay does not re-roll the model at all). max_steps/max_bytes bound the file on disk."""

	path: str | Path
	mode: str = 'record'
	seed: int | None = None
	branch_path: str | Path | None = None
	max_steps: int = TRAJ_MAX_STEPS
	max_bytes: int = TRAJ_MAX_BYTES

	def __post_init__(self) -> None:
		if self.mode not in ('record', 'replay', 'replay-branch'):
			raise ValueError("trajectory mode must be 'record', 'replay' or 'replay-branch'")
		if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
			raise TypeError('trajectory seed must be an int or None')
		self.path = Path(self.path)


def build_recorder(
	config: TrajectoryConfig, *, task: str, model_name: str, secrets: list[tuple[str, str, str | None]]
) -> 'TrajectoryRecorder':
	"""The recorder for a live run: a fresh bounded store, every serialized line redacted with the run's secrets."""
	store = TrajectoryStore(
		path=Path(config.path),
		forms=secret_forms(secrets),
		max_steps=config.max_steps,
		max_bytes=config.max_bytes,
		seed=config.seed,
	)
	store.write_header(task=task, model_name=model_name, seed=config.seed, secret_names=[n for n, _, _ in secrets])
	return TrajectoryRecorder(store=store)


def build_replay(
	config: TrajectoryConfig, *, model_name: str, secrets: list[tuple[str, str, str | None]]
) -> tuple['ReplayController', Any]:
	"""Load a trajectory and build its replay controller + model wrapper. A damaged or foreign trajectory raises
	TrajectoryError, which the Agent turns into a plain live run (never a crash, never a wrong action)."""
	store = TrajectoryStore.load(Path(config.path), forms=secret_forms(secrets))
	controller = ReplayController(store=store, diff=ReplayDiff(corrupt_lines=store.skipped_lines), model_name=model_name)
	ok, why = controller.family_ok()
	if not ok:
		raise TrajectoryError(why or 'model family mismatch')
	return controller, make_replay_model  # the wrapper is constructed around the live model in the agent


class TrajectoryError(Exception):
	"""The trajectory file cannot be read at all (missing header, wrong format, unreadable file)."""


@dataclass
class StepRecord:
	"""One recorded model step. `response` is the ModelResponse as pydantic-ai built it (JSONable form); `cells` are the
	python tool calls that response emitted, in execution order, each with its page evidence and REPL result."""

	step: int
	response: dict[str, Any]
	model_name: str
	cells: list[dict[str, Any]] = field(default_factory=list)
	ts: float = field(default_factory=time.time)

	def to_line(self, forms: list[tuple[str, str]]) -> str:
		line = json.dumps(
			{
				'type': 'step',
				'step': self.step,
				'model_name': self.model_name,
				'response': self.response,
				'cells': self.cells,
				'ts': self.ts,
			},
			ensure_ascii=False,
		)
		return redact_forms(line, forms)


@dataclass
class TrajectoryStore:
	"""Append-only JSONL trajectory with a bounded header + steps and a corrupt-line-tolerant reader.

	Layout: one JSON object per line. Line 1 is the header (format, task hash, model, seed, secret names, bounds); the
	following lines are step records and optional branch markers. Writes are redacted with the run's secret_forms()
	before they touch disk, and fsynced per step so a killed process leaves a valid prefix."""

	path: Path
	forms: list[tuple[str, str]] = field(default_factory=list)
	max_steps: int = TRAJ_MAX_STEPS
	max_bytes: int = TRAJ_MAX_BYTES
	seed: int | None = None
	header: dict[str, Any] = field(default_factory=dict)
	steps: list[StepRecord] = field(default_factory=list)
	skipped_lines: int = 0  # malformed lines tolerated while reading (surfaced in the replay diff)
	closed: bool = False

	# ---------------------------------------------------------------- writing
	def write_header(self, *, task: str, model_name: str, seed: int | None, secret_names: list[str]) -> None:
		self.path.parent.mkdir(parents=True, exist_ok=True)
		self.seed = seed
		self.header = {
			'type': 'header',
			'format': TRAJ_FORMAT,
			'task_digest': _digest(task),
			'model_name': model_name,
			'model_family': _model_family(model_name),
			'seed': seed,
			'secret_names': sorted(secret_names),
			'created_ts': time.time(),
		}
		self._write_line(self.header)

	def record_response(self, step: int, response: 'ModelResponse') -> StepRecord:
		record = StepRecord(
			step=step,
			response=to_jsonable_python(response, fallback=str),
			model_name=getattr(response, 'model_name', '') or '',
		)
		self.steps.append(record)
		self._write_line(json.loads(record.to_line(self.forms)))
		self._enforce_bounds()
		return record

	def record_cell(
		self,
		record: StepRecord,
		*,
		tool_call_id: str,
		code: str,
		selectors: list[str],
		before: dict[str, Any],
		after: dict[str, Any],
		result: dict[str, Any],
	) -> None:
		cell = {
			'tool_call_id': tool_call_id,
			'code': code,
			'selectors': selectors,
			'before_digest': snapshot_digest(before),
			'after_digest': snapshot_digest(after),
			'before': clip_snapshot(before),
			'after': clip_snapshot(after),
			'result': _clip_result(result),
		}
		record.cells.append(cell)
		# Rewrite the step line in place (the last written line): one line per step keeps the file and the index aligned.
		self._rewrite_last_step(record)

	def mark_branch(self, step: int, reason: str) -> None:
		"""The run left the recorded path at `step`; everything afterwards is live decision-making."""
		self._write_line({'type': 'branch', 'step': step, 'reason': reason, 'ts': time.time()})

	def mark_finish(self, status: str) -> None:
		self._write_line({'type': 'finish', 'status': status, 'ts': time.time()})

	def _write_line(self, obj: dict[str, Any]) -> None:
		text = json.dumps(obj, ensure_ascii=False, default=str)
		text = redact_forms(text, self.forms)
		# Defense in depth: even if a forms entry missed an encoding, a bare <secret> value must never be written raw.
		with open(self.path, 'a', encoding='utf-8') as f:
			f.write(text + '\n')
			f.flush()
			os.fsync(f.fileno())

	def _rewrite_last_step(self, record: StepRecord) -> None:
		"""Replace the file's final step line with the record's updated form. Steps are appended in order, so the last
		step line is the last line while a run is recording."""
		lines = self.path.read_text(encoding='utf-8').splitlines()
		lines[-1] = record.to_line(self.forms)
		tmp = self.path.with_suffix('.jsonl.tmp')
		with open(tmp, 'w', encoding='utf-8') as f:
			f.write('\n'.join(lines) + '\n')
			f.flush()
			os.fsync(f.fileno())
		tmp.replace(self.path)
		self._enforce_bounds()

	def _enforce_bounds(self) -> None:
		"""Keep the header, drop the oldest step lines until both bounds hold. The most recent steps (what a replay from
		the middle needs) survive; a truncated trajectory replays its surviving prefix then branches to live."""
		try:
			size = self.path.stat().st_size
			lines = self.path.read_text(encoding='utf-8').splitlines()
		except OSError:
			return

		def is_step(line: str) -> bool:
			try:
				return json.loads(line).get('type') == 'step'
			except json.JSONDecodeError:
				return False

		if size <= self.max_bytes and sum(1 for line in lines if is_step(line)) <= self.max_steps:
			return
		header, *rest = lines
		kept: list[str] = []
		step_count = 0
		for line in reversed(rest):
			if is_step(line):
				step_count += 1
			kept.append(line)
			if step_count > self.max_steps:
				kept.pop()
				break
			if len(header) + sum(len(kept_line) + 1 for kept_line in reversed(kept)) > self.max_bytes:
				kept.pop()
				break
		tmp = self.path.with_suffix('.jsonl.tmp')
		with open(tmp, 'w', encoding='utf-8') as f:
			f.write(header + '\n')
			f.write('\n'.join(reversed(kept)) + ('\n' if kept else ''))
			f.flush()
			os.fsync(f.fileno())
		tmp.replace(self.path)
		self.steps = self.steps[-len(kept) :]

	# ---------------------------------------------------------------- reading
	@classmethod
	def load(cls, path: str | Path, forms: list[tuple[str, str]] | None = None) -> 'TrajectoryStore':
		p = Path(path)
		store = cls(path=p, forms=forms or [])
		if not p.exists():
			raise TrajectoryError(f'trajectory not found: {p}')
		try:
			raw = p.read_text(encoding='utf-8')
		except OSError as e:
			raise TrajectoryError(f'trajectory unreadable: {p}: {e}') from e
		lines = raw.splitlines()
		if not lines:
			raise TrajectoryError(f'trajectory is empty: {p}')
		try:
			header = json.loads(lines[0])
		except json.JSONDecodeError as e:
			raise TrajectoryError(f'trajectory header is not JSON: {p}: {e}') from e
		if not isinstance(header, dict) or header.get('format') != TRAJ_FORMAT or header.get('type') != 'header':
			raise TrajectoryError(f'not a {TRAJ_FORMAT} trajectory: {p}')
		store.header = header
		store.seed = header.get('seed')
		for line in lines[1:]:
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				store.skipped_lines += 1  # a torn append: keep the valid prefix, report the damage at replay time
				continue
			if obj.get('type') != 'step':
				continue  # branch/finish markers are informational for offline tooling
			if not isinstance(obj.get('response'), dict):
				store.skipped_lines += 1  # a structurally damaged step record: count it, never feed it to replay
				continue
			try:
				store.steps.append(
					StepRecord(
						step=int(obj['step']),
						response=obj['response'],
						model_name=obj.get('model_name', ''),
						cells=obj.get('cells', []),
						ts=obj.get('ts', 0.0),
					)
				)
			except (KeyError, TypeError, ValueError):
				store.skipped_lines += 1
		if not store.steps:
			raise TrajectoryError(f'trajectory has no usable step records: {p}')
		return store

	@property
	def secret_names(self) -> list[str]:
		return list(self.header.get('secret_names', []))


def _clip_result(result: dict[str, Any]) -> dict[str, Any]:
	"""What a tool result keeps on disk: text fields cut to RESULT_MAX_CHARS; images dropped entirely (binaries do not
	belong in a trajectory); structural flags kept so replay can compare the outcome class."""
	out: dict[str, Any] = {}
	for k, v in result.items():
		if k == 'images':
			out[k] = [{'media_type': img.get('media_type')} for img in v if isinstance(img, dict)]  # presence only
		elif isinstance(v, str) and len(v) > RESULT_MAX_CHARS:
			out[k] = v[:RESULT_MAX_CHARS] + '\n[trajectory: truncated]'
		else:
			out[k] = v
	return out


# ---------------------------------------------------------------- replay
@dataclass
class TrajectoryRecorder:
	"""The recording side of a live run. The loop's capability feeds it every ModelResponse (record_response) and the
	python() tool feeds it the page evidence and result around each cell (before_cell/after_cell)."""

	store: TrajectoryStore
	_current: StepRecord | None = None
	_pending: dict[str, Any] | None = None  # the before-cell evidence waiting for its result

	def record_response(self, step: int, response: 'ModelResponse') -> None:
		self._current = self.store.record_response(step, response)

	def before_cell(self, code: str, before: dict[str, Any], selectors: list[str]) -> None:
		self._pending = {'code': code, 'before': before, 'selectors': selectors}

	def after_cell(self, after: dict[str, Any], result: dict[str, Any]) -> None:
		if self._current is None or self._pending is None:
			return
		self.store.record_cell(
			self._current,
			tool_call_id='',  # the cell's code is the identity; pydantic-ai's call id is not stable across runs
			code=self._pending['code'],
			selectors=self._pending['selectors'],
			before=self._pending['before'],
			after=after,
			result=result,
		)
		self._pending = None

	def finish(self, status: str) -> None:
		self.store.mark_finish(status)
		self.store.closed = True


@dataclass
class ReplayDiff:
	"""Divergence metrics of one replay run against its source trajectory."""

	replayed_steps: int = 0
	replayed_cells: int = 0
	live_steps: int = 0  # decisions taken by the live model after the branch point
	page_mismatches: int = 0  # snapshot digests that differed before a cell
	selector_mismatches: list[dict[str, Any]] = field(default_factory=list)  # {step, selector, resolved}
	result_mismatches: int = 0  # replayed cells whose REPL result differed in kind from the recorded one
	model_mismatches: list[str] = field(default_factory=list)
	corrupt_lines: int = 0
	branched: bool = False
	branch_step: int | None = None
	branch_reason: str | None = None
	completed: bool = False

	def as_dict(self) -> dict[str, Any]:
		return {k: v for k, v in self.__dict__.items()}

	def summary(self) -> str:
		head = (
			f'replay: {self.replayed_steps} step(s) / {self.replayed_cells} cell(s) replayed deterministically'
			f', {self.live_steps} live step(s)'
		)
		if not self.branched:
			return head + ', no branch'
		return (
			f'{head}, branched at step {self.branch_step} ({self.branch_reason}); page mismatches='
			f'{self.page_mismatches}, stale selectors={len(self.selector_mismatches)}, result mismatches='
			f'{self.result_mismatches}, corrupt lines={self.corrupt_lines}'
		)


@dataclass
class ReplayController:
	"""The state machine a replay run shares between its scripted model and the agent loop's recorder hooks.

	Lifecycle: next_response() serves recorded ModelResponses in order; the loop calls observe_cell() around each cell
	with the live page evidence; the first failed check flips the controller to live mode, after which next_response()
	returns None (the wrapper delegates to the real model) and the run continues as a live branch. Cell lookups use the
	last served record, not pydantic-ai's run_step, so a retried request can never misalign the index."""

	store: TrajectoryStore
	diff: ReplayDiff
	model_name: str
	index: int = 0  # number of recorded responses served (also the position of `_current`)
	live: bool = False
	_live_model: Any = None
	branch_path: Path | None = None
	_current: StepRecord | None = None
	_matched_cells: set[str] = field(default_factory=set)

	@property
	def steps(self) -> list[StepRecord]:
		return self.store.steps

	def family_ok(self) -> tuple[bool, str | None]:
		"""A recorded decision from another model family is not replayed: providers differ in tool-call shapes and the
		recorded args may be invalid for this run. Same-family, different snapshot is allowed and reported. The family
		is the model name's first segment, with the gateway's 'browser-use' treated as its underlying provider."""
		recorded = _model_family(self.store.header.get('model_name', ''))
		current = _model_family(self.model_name)
		if recorded and recorded != current:
			return False, f'model family differs: trajectory={recorded} run={current}'
		return True, None

	def next_response(self) -> 'ModelResponse | None':
		"""The recorded response for the next request, or None when the replay has ended/branched (caller goes live)."""
		if self.live or self.index >= len(self.steps):
			self._enter_live('trajectory exhausted' if not self.live else 'already live')
			return None
		self._current = self.steps[self.index]
		self.index += 1
		self.diff.replayed_steps += 1
		from pydantic_ai.messages import ModelMessagesTypeAdapter

		(resp,) = ModelMessagesTypeAdapter.validate_python([self._current.response])
		return resp  # type: ignore[return-value]

	def _cell(self, code: str) -> dict[str, Any] | None:
		"""The recorded cell for this code in the currently served step. A cell retried within the step (ModelRetry)
		matches its same record again; when no response has been served yet (a direct controller call) fall back to the
		first recorded step rather than failing the whole replay."""
		record = self._current or (self.steps[0] if self.steps else None)
		if record is None:
			return None
		return next((c for c in record.cells if c['code'] == code), None)

	def observe_cell(self, *, step: int, code: str, before: dict[str, Any], selectors_resolved: dict[str, bool | None]) -> bool:
		"""Check the live page against the recorded evidence *before* the cell runs. A changed snapshot digest or a
		selector that provably no longer resolves ends replay: the caller must not execute this recorded cell on a
		different page. A failed probe (None) is 'unknown', never stale."""
		if self.live:
			return True
		cell = self._cell(code)
		if cell is None:  # the code about to run is not on the trajectory: cannot trust it
			self._enter_live(f'cell not in trajectory at request step {step}')
			return False
		if cell['before_digest'] != snapshot_digest(clip_snapshot(before)):
			self.diff.page_mismatches += 1
			self._enter_live(f'page changed before request step {step}')
			return False
		stale = [
			{'step': step, 'selector': sel, 'resolved': selectors_resolved.get(sel)}
			for sel in cell.get('selectors', [])
			if selectors_resolved.get(sel) is False
		]
		if stale:
			self.diff.selector_mismatches.extend(stale)
			self._enter_live(f'selector stale at request step {step}: {stale[0]["selector"][:80]}')
			return False
		return True

	def observe_result(self, code: str, result: dict[str, Any], after: dict[str, Any]) -> None:
		"""After a replayed cell ran on a matching page: count an outcome-class difference (informational) and an
		after-page digest difference (the action landed elsewhere). The latter also ends replay."""
		if self.live:
			return
		cell = self._cell(code)
		if cell is None:
			return
		if code not in self._matched_cells:
			self._matched_cells.add(code)
			self.diff.replayed_cells += 1
		if (result.get('error_type') or None) != ((cell.get('result') or {}).get('error_type') or None):
			self.diff.result_mismatches += 1  # informational: the action still ran on a matching page
		if cell['after_digest'] != snapshot_digest(clip_snapshot(after)):
			self.diff.page_mismatches += 1
			self._enter_live(f'page diverged after request step {self.index}')

	def _enter_live(self, reason: str) -> None:
		if self.live:
			return
		self.live = True
		self.diff.branched, self.diff.branch_reason = True, reason
		self.diff.branch_step = self.index
		self.diff.corrupt_lines = self.store.skipped_lines
		logger.warning('Trajectory replay fell back to live decisions: %s', reason)

	def mark_completed(self) -> None:
		self.diff.completed = not self.live and self.index >= len(self.steps)
		self.diff.corrupt_lines = self.store.skipped_lines


def assert_no_raw_secret(text: str, secret_values: list[str]) -> None:
	"""Final serialization guard used by tests and the writer: a raw secret value (not its <secret:name> placeholder)
	must never appear in a serialized trajectory."""
	for value in secret_values:
		if value and len(value) >= 4 and value in text:
			raise AssertionError(f'trajectory serialization contains a secret value ({len(value)} chars)')


def seed_generators(seed: int | None) -> None:
	"""Best-effort deterministic seeding for the parts of a run Python controls (random; numpy when present). Model
	sampling is decided provider-side and cannot be seeded from here, which is why replay serves recorded decisions
	directly instead of re-rolling the model. None leaves every generator untouched."""
	if seed is None:
		return
	random.seed(seed)
	try:
		import numpy as np  # pyright: ignore[reportMissingImports] - optional dependency, present only in some environments

		np.random.seed(seed & 0xFFFFFFFF)
	except ImportError:
		pass


def make_replay_model(wrapped: Any, controller: 'ReplayController') -> Any:
	"""A WrapperModel that serves recorded ModelResponses while the ReplayController says the page matches, and hands
	every request after the branch point to the real (live) model. Built lazily so importing this module never requires
	pydantic-ai's model internals."""
	from pydantic_ai.models.wrapper import WrapperModel

	class _ReplayModel(WrapperModel):
		async def request(self, messages, model_settings, model_request_parameters):
			response = controller.next_response()
			if response is None:  # branched or trajectory exhausted: the live model decides
				controller.diff.live_steps += 1
				return await self.wrapped.request(messages, model_settings, model_request_parameters)
			return response

	return _ReplayModel(wrapped)
