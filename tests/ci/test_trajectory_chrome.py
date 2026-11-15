"""End-to-end trajectory recording/replay against a real isolated headless Chrome (marked `chrome`).

These cover the full public Agent(trajectory=...) path: a recorded run on one data: page, a deterministic replay on a
fresh browser, a replay where the page changed (must branch to live decisions and withhold the recorded action), and
secret redaction inside the on-disk trajectory. The model is always a scripted FunctionModel: no provider calls.
"""

import json

import pytest
from conftest import script
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from browser_use import Agent, Browser, TrajectoryConfig
from browser_use.trajectory import TrajectoryError, TrajectoryStore

PAGE_A = "data:text/html,<title>Trajectory A</title><h1 id='heading'>Hello A</h1><input id='email' type='text'>"
PAGE_B = "data:text/html,<title>Trajectory B</title><h1 id='heading'>Hello B</h1><div id='other'>different</div>"


def recorded_run_turns(url: str):
	# Cell 1 only navigates (no element selector: the replay starts on about:blank, which legitimately differs). Cell 2
	# reads an element, so its recorded selector and the post-navigation snapshot are what replay validates.
	return [
		('python', {'code': f"new_tab({url!r})\nwait_for_load()\npage = page_info()\npage['title']"}),
		('python', {'code': 'heading = js("document.querySelector(\'#heading\').textContent")\nheading'}),
		('finish_from_repl', {'success': True, 'expression': 'heading'}),
	]


async def test_record_then_replay_is_deterministic(chrome, tmp_path):
	traj = tmp_path / 'run.jsonl'
	model, _ = script(*recorded_run_turns(PAGE_A))
	browser = Browser(cdp_url=chrome['http'])
	async with Agent(
		'Read the page',
		model,
		browser=browser,
		workspace=tmp_path / 'w1',
		trajectory=TrajectoryConfig(traj, seed=99),
		judge=False,
	) as agent:
		first = await agent.run()
	assert first.final_result() == 'Hello A', first.error
	store = TrajectoryStore.load(traj)
	assert store.header['seed'] == 99 and store.header['format'] == 'browser-use-trajectory/1'
	assert [s.step for s in store.steps] == [1, 2, 3]
	cells = [c for s in store.steps for c in s.cells]
	assert cells and all(c['before_digest'] and c['after_digest'] for c in cells)
	assert any('#heading' in c['selectors'] for c in cells)

	# Replay into a fresh browser/workspace from the same page: recorded decisions, no live model turn.
	live_calls = []

	def live_only(messages, info: AgentInfo):
		live_calls.append(1)
		raise AssertionError('a clean replay must not consult the live model')

	replay_model = FunctionModel(live_only)
	async with Agent(
		'Read the page',
		replay_model,
		browser=Browser(cdp_url=chrome['http']),
		workspace=tmp_path / 'w2',
		trajectory=TrajectoryConfig(traj, mode='replay'),
		judge=False,
	) as agent2:
		second = await agent2.run()
	assert second.final_result() == 'Hello A', second.error
	assert not live_calls
	diff = agent2.replay_diff
	assert diff is not None and diff['branched'] is False and diff['completed'] is True
	assert diff['replayed_steps'] == 3 and diff['replayed_cells'] >= 2 and diff['live_steps'] == 0


async def _open_page(chrome, workspace, url):
	"""Drive a browser to `url` and leave it there, so the next Agent attaches to an already-changed page."""
	from browser_use.repl import Repl

	browser = Browser(cdp_url=chrome['http'], keep_alive=True)
	env = await browser.start()
	repl = Repl(workspace, env)
	await repl.start()
	try:
		await repl.call('exec', f'new_tab({url!r})\nwait_for_load()', 30)
	finally:
		await repl.close()
	return browser


async def test_replay_branches_to_live_when_page_changed(chrome, tmp_path):
	traj = tmp_path / 'run.jsonl'
	model, _ = script(*recorded_run_turns(PAGE_A))
	async with Agent(
		'Read the page',
		model,
		browser=Browser(cdp_url=chrome['http']),
		workspace=tmp_path / 'w1',
		trajectory=TrajectoryConfig(traj),
		judge=False,
	) as agent:
		assert (await agent.run()).final_result() == 'Hello A'

	# The page changed since recording: the browser now shows PAGE_B. Replay must detect the mismatch, withhold the
	# recorded cell (it would read the wrong page) and hand the turn to the live model.
	changed = await _open_page(chrome, tmp_path / 'w2', PAGE_B)

	def fn(messages, info: AgentInfo):
		return ModelResponse(parts=[ToolCallPart('finish', {'success': False, 'result': 'live recovery'})])

	try:
		async with Agent(
			'Read the page',
			FunctionModel(fn),
			browser=changed,
			workspace=tmp_path / 'w2',
			trajectory=TrajectoryConfig(traj, mode='replay'),
			judge=False,
		) as agent2:
			result = await agent2.run()
	finally:
		await changed.stop()
	diff = agent2.replay_diff
	assert result.final_result() == 'live recovery'
	assert diff['branched'] is True and diff['page_mismatches'] >= 1
	assert diff['branch_reason'] and 'page changed' in diff['branch_reason']
	assert diff['live_steps'] >= 1


async def test_replay_branch_writes_a_child_trajectory(chrome, tmp_path):
	traj = tmp_path / 'run.jsonl'
	branch = tmp_path / 'child.jsonl'
	model, _ = script(*recorded_run_turns(PAGE_A))
	async with Agent(
		'Read the page',
		model,
		browser=Browser(cdp_url=chrome['http']),
		workspace=tmp_path / 'w1',
		trajectory=TrajectoryConfig(traj),
		judge=False,
	) as agent:
		await agent.run()

	changed = await _open_page(chrome, tmp_path / 'w2', PAGE_B)

	def fn(messages, info):
		return ModelResponse(parts=[ToolCallPart('finish', {'success': True, 'result': 'branched answer'})])

	try:
		async with Agent(
			'Read the page',
			FunctionModel(fn),
			browser=changed,
			workspace=tmp_path / 'w2',
			trajectory=TrajectoryConfig(traj, mode='replay-branch', branch_path=branch),
			judge=False,
		) as agent2:
			result = await agent2.run()
	finally:
		await changed.stop()
	assert result.final_result() == 'branched answer'
	assert branch.exists()
	child = TrajectoryStore.load(branch)
	assert child.header['format'] == 'browser-use-trajectory/1'
	assert any(json.loads(line).get('type') == 'branch' for line in branch.read_text().splitlines()[1:])


async def test_trajectory_on_disk_never_contains_the_secret(chrome, tmp_path):
	secret = 'hunter2-Zq9'
	traj = tmp_path / 'secret.jsonl'
	turns = [
		(
			'python',
			{
				'code': f"new_tab('data:text/html,<title>Secret</title><input id=pw>')\nwait_for_load()\n"
				f"type_secret('pw')\nprint('the value is {secret}')\npage_info()['title']"
			},
		),
		('finish_from_repl', {'success': True, 'expression': "page_info()['title']"}),
	]
	model, _ = script(*turns)
	async with Agent(
		'Log in',
		model,
		browser=Browser(cdp_url=chrome['http']),
		workspace=tmp_path / 'w',
		sensitive_data={'pw': secret},
		trajectory=TrajectoryConfig(traj),
		judge=False,
	) as agent:
		result = await agent.run()
	assert result.final_result() == 'Secret'
	raw = traj.read_text(encoding='utf-8')
	assert secret not in raw  # neither the printed cell value nor its args may serialize the raw secret
	assert '<secret:pw>' in raw
	assert store_has_no_image_binary(traj)


def store_has_no_image_binary(path) -> bool:
	store = TrajectoryStore.load(path)
	for step in store.steps:
		for cell in step.cells:
			for image in cell['result'].get('images', []):
				if 'data' in image:  # only the media_type presence is kept, never bytes
					return False
	return True


async def test_missing_trajectory_degrades_to_a_live_run(chrome, tmp_path):
	missing = tmp_path / 'absent.jsonl'
	model, _ = script(('finish', {'success': True, 'result': 'live anyway'}))
	async with Agent(
		'task',
		model,
		browser=Browser(cdp_url=chrome['http']),
		workspace=tmp_path / 'w',
		trajectory=TrajectoryConfig(missing, mode='replay'),
		judge=False,
	) as agent:
		result = await agent.run()  # must not raise: an unreadable trajectory is a live run, not a crash
	assert result.final_result() == 'live anyway'
	assert agent.replay_diff is None  # replay never started
	assert not missing.exists()


async def test_a_bare_path_argument_records(chrome, tmp_path):
	"""Agent(trajectory='path.jsonl') is shorthand for TrajectoryConfig('path.jsonl', mode='record')."""
	traj = tmp_path / 'shorthand.jsonl'
	model, _ = script(('finish', {'success': True, 'result': 'ok'}))
	async with Agent(
		't', model, browser=Browser(cdp_url=chrome['http']), workspace=tmp_path / 'w', trajectory=str(traj), judge=False
	) as agent:
		result = await agent.run()
	assert result.final_result() == 'ok'
	assert traj.exists() and TrajectoryStore.load(traj).header['format'] == 'browser-use-trajectory/1'


def test_trajectory_option_type_is_validated():
	with pytest.raises(TypeError):
		Agent('t', None, trajectory=123)  # type: ignore[arg-type]
	with pytest.raises(TrajectoryError):
		TrajectoryStore.load('/nonexistent/path/for/validation.jsonl')
