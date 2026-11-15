"""Trajectory recording and replay (browser_use/trajectory.py and its loop.py hooks).

Two layers:
- tests without Chrome drive the real agent loop with a ScriptedRepl double that answers the trajectory's snapshot /
  selector probes itself, so recording, replay, branching, corruption and redaction are deterministic here;
- tests marked `chrome` (test_trajectory_chrome below lives in this file too) run a real headless Chromium end to end.
Nothing here calls a model provider: every model is a FunctionModel script or a recorded response.
"""

import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from browser_use.agent.loop import Session, build_agent, run
from browser_use.repl.session import scoped_secrets, secret_forms
from browser_use.trajectory import (
	ReplayController,
	ReplayDiff,
	TrajectoryConfig,
	TrajectoryError,
	TrajectoryRecorder,
	TrajectoryStore,
	build_recorder,
	build_replay,
	extract_selectors,
	snapshot_digest,
)

SECRET_VALUE = 'hunter2-Zq9'
SNAPSHOT_MARK = 'dumps(js('  # the _page_snapshot probe wraps its JS in js(...) and json-dumps the result


class ScriptedRepl:
	"""A Repl double for the trajectory tests: answers snapshot/selector probes with controllable page state, and every
	other cell with a scripted per-code result. Records the model cells that actually executed."""

	def __init__(self, workspace: Path, secrets=None, *, page: dict | None = None, selectors=None, results=None):
		self.workspace = self.state = workspace
		self.secrets = scoped_secrets(secrets)
		self.search = None
		self.mode = 'default'
		self.start_ns = 0
		self._page = page or {
			'url': 'https://site.test/',
			'title': 'Login',
			'ready': 'complete',
			'element_count': 3,
			'inputs': ['input[id=email]'],
			'text_head': 'sign in',
		}
		# default: every recorded selector resolves; a stale-selector case passes its own map with a False entry
		self._selectors = selectors if selectors is not None else {'#email': True, '.missing': True}
		self._results = results or {}
		self.executed: list[str] = []
		self.n = 0

	def with_page(self, page: dict) -> 'ScriptedRepl':
		self._page = page
		return self

	async def call(self, op, code, timeout):
		if SNAPSHOT_MARK in code:  # the snapshot probe from _page_snapshot
			return {'output': json.dumps(self._page), 'error': None, 'images': [], 'error_type': None}
		if '__bu_selector_probe' in code:  # the selector-resolution probe from _selectors_resolve
			import ast

			assigned = next(line for line in code.splitlines() if line.startswith('_sel = '))
			payload = ast.literal_eval(assigned.removeprefix('_sel = '))
			return {
				'output': json.dumps({s: self._selectors.get(s) for s in payload}),
				'error': None,
				'images': [],
				'error_type': None,
			}
		self.n += 1
		self.executed.append(code)
		planned = self._results.get(code, {'output': f'ran {self.n}', 'error': None, 'error_type': None})
		return {'images': [], **planned}


def model_turns():
	"""Three scripted turns: two python cells then finish."""
	return [
		('python', {'code': "wait_for_element('#email')\nfill_input('#email', 'a@b.c')"}),
		('python', {'code': 'js("document.querySelector(\'.missing\')")'}),
		('finish', {'success': True, 'result': 'done'}),
	]


def scripted_model(turns):
	calls = []

	def fn(messages, info: AgentInfo):
		calls.append(len(messages))
		turn = turns[len(calls) - 1]
		items = turn if isinstance(turn, list) else [turn]
		return ModelResponse(parts=[ToolCallPart(n, a) for n, a in items])

	return FunctionModel(fn), calls


async def record_run(tmp_path, repl, *, secrets=None, seed=None):
	"""Run the loop once with recording on; return (result, store path)."""
	path = tmp_path / 'traj.jsonl'
	model, _ = scripted_model(model_turns())
	store = TrajectoryStore(path=path, forms=secret_forms(scoped_secrets(secrets)), seed=seed)
	store.write_header(
		task='the task', model_name='function:x', seed=seed, secret_names=[n for n, _, _ in scoped_secrets(secrets)]
	)
	session = Session(repl=repl, trajectory=TrajectoryRecorder(store))
	agent = build_agent(model)
	try:
		result = await run(agent, 'the task', session, 'function:x')
		session.trajectory.finish(result.status)
		return result, path
	except BaseException:
		session.trajectory.finish('error')
		raise


# ------------------------------------------------------------- pure unit tests
def test_extract_selectors_finds_locators_not_text():
	code = (
		'js("document.querySelector(\'#a .b\')")\nwait_for_element("div[x=\'y\']")\n'
		"fill_input('#email', 'typed text')\ncdp('DOM.querySelector', selector='.z')\ngoto_url('https://h.test/p')"
	)
	sels = extract_selectors(code)
	assert sels == ['#a .b', "div[x='y']", '#email', '.z']
	assert extract_selectors("goto_url('https://h.test/p')\nprint('plain words')") == []
	assert extract_selectors('') == []


def test_config_validation():
	cfg = TrajectoryConfig('t.jsonl')
	assert cfg.mode == 'record' and cfg.seed is None
	assert TrajectoryConfig('t.jsonl', mode='replay-branch', seed=3).seed == 3
	with pytest.raises(ValueError):
		TrajectoryConfig('t.jsonl', mode='rewind')
	with pytest.raises(TypeError):
		TrajectoryConfig('t.jsonl', seed=1.5)
	with pytest.raises(TypeError):
		TrajectoryConfig('t.jsonl', seed=True)


def test_record_and_load_roundtrip(tmp_path):
	path = tmp_path / 't.jsonl'
	forms = secret_forms(scoped_secrets({'pw': SECRET_VALUE}))
	recorder = build_recorder(
		TrajectoryConfig(path), task='task', model_name='function:x', secrets=scoped_secrets({'pw': SECRET_VALUE})
	)
	snap = {'url': 'u', 'title': 'T', 'text_head': SECRET_VALUE}
	resp = ModelResponse(parts=[ToolCallPart('python', {'code': f'x = "{SECRET_VALUE}"'}, tool_call_id='c1')])
	recorder.record_response(1, resp)
	recorder.before_cell(resp.parts[0].args_as_dict()['code'], snap, ['#email'])
	recorder.after_cell(
		snap, {'error_type': None, 'output': SECRET_VALUE, 'images': [{'media_type': 'image/png', 'data': 'BIN'}]}
	)
	recorder.finish('completed')
	raw = path.read_text(encoding='utf-8')
	assert SECRET_VALUE not in raw  # serialization never contains the raw secret
	assert 'BIN' not in raw  # image binaries are not stored
	assert '<secret:pw>' in raw
	loaded = TrajectoryStore.load(path, forms=forms)
	assert loaded.header['format'] == 'browser-use-trajectory/1'
	assert loaded.header['secret_names'] == ['pw']
	assert len(loaded.steps) == 1
	cell = loaded.steps[0].cells[0]
	assert cell['selectors'] == ['#email']
	assert cell['before_digest'] == snapshot_digest(snap)
	# the recorded response deserializes back into pydantic-ai messages
	from pydantic_ai.messages import ModelMessagesTypeAdapter

	(restored,) = ModelMessagesTypeAdapter.validate_python([loaded.steps[0].response])
	assert isinstance(restored.parts[0], ToolCallPart)


def test_corrupt_header_and_torn_lines_are_reported_not_fatal(tmp_path):
	good = tmp_path / 'good.jsonl'
	forms = secret_forms(scoped_secrets({'pw': SECRET_VALUE}))
	store = TrajectoryStore(path=good, forms=forms)
	store.write_header(task='t', model_name='function:x', seed=None, secret_names=[])
	store.record_response(1, ModelResponse(parts=[TextPart('ok')]))
	# torn append + structurally invalid step
	with open(good, 'a', encoding='utf-8') as f:
		f.write('{"type":"step", broken\n')
		f.write('{"type":"step","step":99}\n')
	loaded = TrajectoryStore.load(good, forms=forms)
	assert loaded.skipped_lines == 2 and len(loaded.steps) == 1
	for name, content in [
		('bad.jsonl', 'not json\n'),
		('empty.jsonl', ''),
		('foreign.jsonl', json.dumps({'type': 'header', 'format': 'something-else'}) + '\n'),
	]:
		p = tmp_path / name
		p.write_text(content)
		with pytest.raises(TrajectoryError):
			TrajectoryStore.load(p)


def test_storage_bounds_drop_oldest_keep_header_and_tail(tmp_path):
	path = tmp_path / 'bounded.jsonl'
	store = TrajectoryStore(path=path, forms=[], max_steps=4, max_bytes=1_000_000)
	store.write_header(task='t', model_name='function:x', seed=None, secret_names=[])
	for i in range(10):
		store.record_response(i + 1, ModelResponse(parts=[TextPart('x' * 100)]))
	loaded = TrajectoryStore.load(path)
	assert len(loaded.steps) == 4  # the newest four
	assert [s.step for s in loaded.steps] == [7, 8, 9, 10]
	assert loaded.header['type'] == 'header'
	# byte bound
	big = tmp_path / 'bytes.jsonl'
	store2 = TrajectoryStore(path=big, forms=[], max_steps=1000, max_bytes=2_000)
	store2.write_header(task='t', model_name='function:x', seed=None, secret_names=[])
	for i in range(40):
		store2.record_response(i + 1, ModelResponse(parts=[TextPart('word ' * 50)]))
	assert big.stat().st_size <= 2_000 + 600
	assert big.read_text(encoding='utf-8').splitlines()[0].startswith('{"type": "header"')


def test_model_family_mismatch_refuses_replay(tmp_path):
	path = tmp_path / 'fam.jsonl'
	store = TrajectoryStore(path=path)
	store.write_header(task='t', model_name='anthropic:claude-x', seed=None, secret_names=[])
	store.record_response(1, ModelResponse(parts=[TextPart('ok')]))
	# build_replay refuses a foreign family up front, so the Agent can degrade to a live run
	with pytest.raises(TrajectoryError, match='model family differs'):
		build_replay(TrajectoryConfig(path, mode='replay'), model_name='function:x', secrets=[])
	# same family constructs the controller
	controller, _ = build_replay(TrajectoryConfig(path, mode='replay'), model_name='anthropic:claude-y', secrets=[])
	ok, why = controller.family_ok()
	assert ok and why is None


# ------------------------------------------------------------- loop integration (no Chrome)
async def test_recording_captures_decisions_cells_and_digests_without_changing_result(tmp_path):
	repl = ScriptedRepl(tmp_path)
	result, path = await record_run(tmp_path, repl)
	assert result.status == 'completed' and result.output == 'done'
	assert len(repl.executed) == 2  # both model cells really ran
	store = TrajectoryStore.load(path)
	assert [s.step for s in store.steps] == [1, 2, 3]
	cells = [c for s in store.steps for c in s.cells]
	assert [c['code'] for c in cells] == [model_turns()[0][1]['code'], model_turns()[1][1]['code']]
	assert all(c['before_digest'] and c['after_digest'] for c in cells)
	assert cells[0]['selectors'] == ['#email']
	assert cells[1]['selectors'] == ['.missing']
	assert all(isinstance(c['result'], dict) and 'images' in c['result'] for c in cells)


async def test_default_run_records_nothing(tmp_path):
	"""The feature is opt-in: a Session without trajectory writes no trajectory and the loop behaves as before."""
	repl = ScriptedRepl(tmp_path)
	model, _ = scripted_model(model_turns())
	session = Session(repl=repl)
	result = await run(build_agent(model), 'task', session, 'function:x')
	assert result.status == 'completed'
	assert not list(tmp_path.glob('*.jsonl'))
	assert session.trajectory is None


async def test_successful_replay_reuses_recorded_decisions(tmp_path):
	_, path = await record_run(tmp_path, ScriptedRepl(tmp_path))
	# A fresh REPL on the same page: replay must serve recorded responses and never ask its (unused) live model.
	repl = ScriptedRepl(tmp_path / 'replay')
	repl.workspace.mkdir(parents=True, exist_ok=True)
	store = TrajectoryStore.load(path)
	controller = ReplayController(store=store, diff=ReplayDiff(), model_name='function:x')

	def live_only(messages, info):  # must never be called while the recorded prefix lasts
		raise AssertionError('live model should not be consulted during a clean replay')

	from browser_use.trajectory import make_replay_model

	model = make_replay_model(FunctionModel(live_only), controller)
	session = Session(repl=repl, trajectory=controller)
	result = await run(build_agent(model), 'the task', session, 'function:x')
	controller.mark_completed()
	assert result.status == 'completed' and result.output == 'done'
	assert controller.diff.completed and not controller.diff.branched
	assert controller.diff.replayed_steps == 3 and controller.diff.replayed_cells == 2
	assert controller.diff.live_steps == 0 and repl.executed  # recorded cells executed against the fresh page


async def test_page_change_branches_to_live_decisions(tmp_path):
	_, path = await record_run(tmp_path, ScriptedRepl(tmp_path))
	changed = {
		'url': 'https://site.test/other',
		'title': 'Different',
		'ready': 'complete',
		'element_count': 99,
		'inputs': [],
		'text_head': 'a totally different page',
	}
	repl = ScriptedRepl(tmp_path / 'replay2', page=changed)
	repl.workspace.mkdir(parents=True, exist_ok=True)
	store = TrajectoryStore.load(path)
	controller = ReplayController(store=store, diff=ReplayDiff(), model_name='function:x')
	live_calls = []

	def fn(messages, info: AgentInfo):
		live_calls.append(1)
		# the first live turn after the branch: deliver, so the run ends cleanly
		return ModelResponse(parts=[ToolCallPart('finish', {'success': False, 'result': 'recovered live'})])

	from browser_use.trajectory import make_replay_model

	model = make_replay_model(FunctionModel(fn), controller)
	session = Session(repl=repl, trajectory=controller)
	result = await run(build_agent(model), 'the task', session, 'function:x')
	controller.mark_completed()
	assert result.output == 'recovered live'
	assert controller.diff.branched and controller.diff.page_mismatches >= 1
	assert controller.diff.branch_reason and 'page changed' in controller.diff.branch_reason
	assert controller.diff.live_steps >= 1 and live_calls
	# the recorded cell was withheld: it never executed on the changed page
	assert repl.executed == []


async def test_stale_selector_branches_without_running_the_cell(tmp_path):
	_, path = await record_run(tmp_path, ScriptedRepl(tmp_path))
	# same page digest, but the recorded selector now fails to resolve
	same_page = {
		'url': 'https://site.test/',
		'title': 'Login',
		'ready': 'complete',
		'element_count': 3,
		'inputs': ['input[id=email]'],
		'text_head': 'sign in',
	}
	repl = ScriptedRepl(tmp_path / 'replay3', page=same_page, selectors={'#email': False, '.missing': False})
	repl.workspace.mkdir(parents=True, exist_ok=True)
	store = TrajectoryStore.load(path)
	controller = ReplayController(store=store, diff=ReplayDiff(), model_name='function:x')

	def fn(messages, info):
		return ModelResponse(parts=[ToolCallPart('finish', {'success': False, 'result': 'live fallback'})])

	from browser_use.trajectory import make_replay_model

	session = Session(repl=repl, trajectory=controller)
	result = await run(build_agent(make_replay_model(FunctionModel(fn), controller)), 'the task', session, 'function:x')
	controller.mark_completed()
	assert result.output == 'live fallback'
	assert controller.diff.branched
	assert any(m['selector'] == '#email' for m in controller.diff.selector_mismatches)
	assert repl.executed == []  # the stale action was not executed


async def test_trajectory_corruption_degrades_to_live(tmp_path):
	"""A trajectory that cannot be loaded must not crash or block the run: build_replay raises, the Agent goes live."""
	missing = tmp_path / 'nope.jsonl'
	with pytest.raises(TrajectoryError):
		build_replay(TrajectoryConfig(missing, mode='replay'), model_name='function:x', secrets=[])
	# and a store whose lines were truncated mid-run still loads its valid prefix
	_, path = await record_run(tmp_path, ScriptedRepl(tmp_path / 'src'))
	lines = path.read_text().splitlines()
	path.write_text('\n'.join(lines[:2]) + '\n{broken\n')
	store = TrajectoryStore.load(path)
	assert store.skipped_lines == 1 and store.steps  # the valid first step survived


async def test_recording_redacts_secrets_everywhere(tmp_path):
	secrets = {'pw': SECRET_VALUE}
	repl = ScriptedRepl(
		tmp_path,
		secrets=secrets,
		results={
			'js("document.querySelector(\'.missing\')")': {
				'output': f'page says {SECRET_VALUE}',
				'error': None,
				'error_type': None,
			}
		},
	)
	_, path = await record_run(tmp_path, repl, secrets=secrets)
	raw = path.read_text(encoding='utf-8')
	assert SECRET_VALUE not in raw
	assert '<secret:pw>' in raw
	# and the loaded, re-serialized form stays redacted
	reloaded = json.dumps(TrajectoryStore.load(path, forms=secret_forms(scoped_secrets(secrets))).header)
	assert SECRET_VALUE not in reloaded


async def test_seed_is_recorded_and_applied(tmp_path):
	import random

	repl = ScriptedRepl(tmp_path)
	await record_run(tmp_path, repl, seed=12345)
	store = TrajectoryStore.load(tmp_path / 'traj.jsonl')
	assert store.header['seed'] == 12345
	from browser_use.trajectory import seed_generators

	seed_generators(12345)
	a = random.random()
	seed_generators(12345)
	b = random.random()
	assert a == b  # deterministic seeding
	seed_generators(None)  # does not raise


async def test_replay_diff_metrics_shape(tmp_path):
	_, path = await record_run(tmp_path, ScriptedRepl(tmp_path))
	store = TrajectoryStore.load(path)
	diff = ReplayDiff(
		replayed_steps=3,
		replayed_cells=2,
		page_mismatches=1,
		branched=True,
		branch_step=1,
		branch_reason='page changed before request step 1',
		corrupt_lines=store.skipped_lines,
	)
	d = diff.as_dict()
	assert {
		'replayed_steps',
		'replayed_cells',
		'live_steps',
		'page_mismatches',
		'selector_mismatches',
		'result_mismatches',
		'branched',
		'branch_step',
		'branch_reason',
		'completed',
		'corrupt_lines',
	} <= set(d)
	assert 'page changed' in diff.summary()
