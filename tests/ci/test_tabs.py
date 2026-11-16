"""Multi-tab/popup orchestration (browser_use/repl/tabs.py): registry bounds, popup budgets, crash/fallback,
download metrics — unit-tested with a fake harness, then end-to-end against a real headless Chrome (daemon fixture)."""

import json
import time

import pytest

from browser_use.repl import Repl
from browser_use.repl.tabs import TabGoneError, TabLimitError, TabManager


class FakeHelpers:
	"""The parts of browser_harness.helpers the TabManager uses, in memory."""

	def __init__(self):
		self.tabs = {'main': {'targetId': 'main', 'url': 'https://a.test/', 'title': 'A'}}
		self.attached = 'main'
		self.next_id = 0
		self.fail_new_tab = False
		self.drained = 0

	def current_tab(self):
		return self.tabs[self.attached]

	def list_tabs(self):
		return list(self.tabs.values())

	def new_tab(self, url='about:blank'):
		if self.fail_new_tab:
			raise RuntimeError('createTarget failed')
		self.next_id += 1
		tid = f't{self.next_id}'
		self.tabs[tid] = {'targetId': tid, 'url': url, 'title': ''}
		self.attached = tid
		return tid

	def switch_tab(self, tid, activate=False):
		if tid not in self.tabs:
			raise RuntimeError(f'No target with given id: {tid}')
		self.attached = tid
		return f'session-{tid}'

	def wait_for_load(self, timeout=15.0):
		return True

	def page_info(self):
		tab = self.tabs[self.attached]
		return {'url': tab['url'], 'title': tab['title']}

	def drain_events(self):
		self.drained += 1
		return []


class FakeCdp:
	def __init__(self):
		self.closed = []

	def __call__(self, method, **params):
		if method == 'Target.closeTarget':
			self.closed.append(params['targetId'])
		return {}


def created(tid, url='https://p.test/', opener=None):
	info = {'targetId': tid, 'type': 'page', 'url': url, 'title': ''}
	if opener:
		info['openerId'] = opener
	return {'method': 'Target.targetCreated', 'params': {'targetInfo': info}}


def manager(helpers=None, **kw):
	helpers = helpers or FakeHelpers()
	cdp = FakeCdp()
	m = TabManager(**kw)
	m.bind(cdp, helpers)
	m.seed()
	return m, helpers, cdp


def test_registry_is_bounded_and_untracked_tabs_are_counted():
	m, helpers, cdp = manager(max_registry=10)
	m.observe([created(f'x{i}') for i in range(50)])
	assert len(m.tabs) <= 10
	assert m.metrics()['untracked'] == 50 - (10 - 1)  # the main tab took one seat


def test_popup_budgets_close_overflow_popups():
	m, helpers, cdp = manager(max_popups_per_page=2, max_popups_total=3)
	m.observe([created(f'p{i}', opener='main') for i in range(3)])  # per-page budget: 2
	assert sorted(cdp.closed) == ['p2']
	m.observe([created('q0', opener='other'), created('q1', opener='other')])  # total budget: 3 live
	assert cdp.closed == ['p2', 'q1']
	stats = m.metrics()
	assert (stats['popups'], stats['popups_rejected']) == (3, 2)
	assert m.popups() and all(p['kind'] == 'popup' for p in m.popups())


def test_await_popup_returns_none_when_blocked_and_the_popup_when_it_comes():
	m, helpers, cdp = manager()
	start = time.monotonic()
	assert m.await_popup(0.3) is None  # a blocked window.open(): an answer, never a hang
	assert time.monotonic() - start < 2
	m.observe([created('p0', opener='main')])
	popup = m.await_popup(0.3)
	assert popup['targetId'] == 'p0' and popup['kind'] == 'popup'


def test_open_tab_limit_raises_and_names_the_limit():
	m, helpers, cdp = manager(max_tabs=2)
	m.open('https://b.test/')
	with pytest.raises(TabLimitError, match='tab limit'):
		m.open('https://c.test/')
	assert m.metrics()['tab_limit_hits'] == 1


def test_a_failed_open_falls_back_to_the_main_tab():
	m, helpers, cdp = manager()
	helpers.fail_new_tab = True
	with pytest.raises(RuntimeError, match='createTarget failed'):
		m.open('https://b.test/')
	assert helpers.attached == 'main'  # back on the main tab, not stuck on a broken one
	assert m.metrics()['fallbacks'] == 1
	assert 'back on the main tab' in m.take_notices()[0]


def test_a_crashed_tab_is_marked_and_refused():
	m, helpers, cdp = manager()
	m.open('https://b.test/')
	crashed = helpers.attached
	m.observe([{'method': 'Target.targetCrashed', 'params': {}}])
	assert m.metrics()['crashes'] == 1
	tab = next(t for t in m.listing() if t['targetId'] == crashed)
	assert tab['crashed'] is True
	with pytest.raises(TabGoneError, match='crashed'):
		m.switch(crashed)
	m.switch('main')  # the main tab still works
	assert helpers.attached == 'main'


def test_closed_tabs_are_deregistered_by_event():
	m, helpers, cdp = manager()
	m.open('https://b.test/')
	tid = helpers.attached
	m.observe([{'method': 'Target.targetDestroyed', 'params': {'targetId': tid}}])
	assert [t['targetId'] for t in m.listing()] == ['main']
	with pytest.raises(TabGoneError, match='closed'):
		m.switch(tid)


def test_completed_downloads_count_once_per_guid():
	m, helpers, cdp = manager()
	event = {'method': 'Browser.downloadProgress', 'params': {'guid': 'g1', 'state': 'completed'}}
	m.observe([event, event, {'method': 'Page.downloadProgress', 'params': {'guid': 'g2', 'state': 'completed'}}])
	assert m.metrics()['downloads_completed'] == 2


def test_cleanup_closes_own_tabs_but_not_external_ones():
	m, helpers, cdp = manager(own_browser=False)
	m.open('https://b.test/')
	m.observe([created('user-tab')])  # the user's own tab: registered, never closed by cleanup
	assert m.cleanup() == 1
	assert sorted(cdp.closed) == ['t1']
	assert [t['targetId'] for t in m.listing()] == ['main', 'user-tab']


def test_metrics_have_the_audit_fields():
	m, helpers, cdp = manager()
	stats = m.metrics()
	for key in (
		'tabs_open',
		'tabs_opened',
		'tabs_closed',
		'popups',
		'popups_rejected',
		'crashes',
		'fallbacks',
		'downloads_completed',
		'tab_limit_hits',
		'peak_tabs',
		'memory_kb',
	):
		assert key in stats, key


def test_events_are_logged_for_audit(tmp_path):
	m, helpers, cdp = manager(log_path=tmp_path / 'tabs.jsonl')
	m.open('https://b.test/')
	m.observe([created('p0', opener='main')])
	records = [json.loads(line) for line in (tmp_path / 'tabs.jsonl').read_text().splitlines()]
	assert [r['event'] for r in records] == ['opened', 'popup']


# --- end-to-end against the isolated headless Chrome (daemon fixture) ---

# The module's Chrome is shared: close what earlier tests left, so each test's registry starts from the main tab.
RESET = (
	'for t in list_tabs():\n'
	"\tif t['targetId'] != current_tab()['targetId']:\n"
	"\t\ttry: close_tab(t['targetId'])\n"
	'\t\texcept Exception: pass\n'
	'import time; time.sleep(0.5)\ndrain_events()\n'
)


@pytest.fixture(scope='module')
def httpd():
	"""A local HTTP server: real URLs (Chrome reports data: URLs as '' in target info, which new_tab treats as blank)."""
	import threading
	from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

	pages = {
		'/a': '<title>A</title><h1>A</h1>',
		'/b': '<title>B</title><h1>B</h1>',
		'/c': '<title>C</title>',
		'/link': '<a href="/pop" target="_blank" style="font-size:60px">OPEN</a>',
		'/pop': '<title>pop</title>',
	}

	class Handler(BaseHTTPRequestHandler):
		def do_GET(self):
			body = pages.get(self.path, '<title>?</title>').encode()
			self.send_response(200)
			self.send_header('Content-Type', 'text/html')
			self.send_header('Content-Length', str(len(body)))
			self.end_headers()
			self.wfile.write(body)

		def log_message(self, *args):
			pass

	server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
	threading.Thread(target=server.serve_forever, daemon=True).start()
	yield f'http://127.0.0.1:{server.server_address[1]}'
	server.shutdown()


async def test_open_tab_registers_waits_and_backfills(daemon, httpd, tmp_path):
	repl = Repl(tmp_path, daemon)
	try:
		await repl.call('exec', RESET, 30)
		# The first open reuses the daemon's blank tab (as new_tab does), the second is a real new tab.
		r = await repl.call('exec', f"open_tab('{httpd}/a')\ninfo = open_tab('{httpd}/b')\nprint(info['title'])", 30)
		assert r['error'] is None, r['error']
		assert 'B' in r['output']
		r = await repl.call('exec', "for t in tabs(): print(t['kind'], t['url'].rsplit('/', 1)[-1])\nprint(tab_stats())", 30)
		assert r['error'] is None, r['error']
		assert 'main a' in r['output'] and 'tab b' in r['output']
		assert "'tabs_opened': 1" in r['output'] and "'tabs_open': 2" in r['output']
		assert "'memory_kb':" in r['output']
	finally:
		await repl.close()


async def test_popups_are_registered_and_the_overflow_closed(daemon, httpd, tmp_path):
	repl = Repl(tmp_path, daemon)
	try:
		await repl.call('exec', RESET, 30)
		# Six genuine clicks, one popup each; MAX_POPUPS_PER_PAGE (4) keeps four and closes the rest as they arrive.
		code = (
			f"goto_url('{httpd}/link')\nwait_for_load()\n"
			'import time\n'
			'for i in range(6):\n\tclick_at_xy(100, 50)\n\ttime.sleep(0.4)\n'
			'time.sleep(1)\ndrain_events()\n'
			"print('popups', len(popups()))\nprint(tab_stats())"
		)
		r = await repl.call('exec', code, 60)
		assert r['error'] is None, r['error']
		assert 'popups 4' in r['output'], r['output']
		assert "'popups_rejected': 2" in r['output']
		assert '[popup limit reached: closed' in r['output']  # the notice the model gets after the cell
		assert '[popup opened:' in r['output']
	finally:
		await repl.close()


async def test_a_crashed_tab_falls_back_to_the_main_tab(daemon, httpd, tmp_path):
	repl = Repl(tmp_path, daemon)
	try:
		await repl.call('exec', RESET, 30)
		# The crashed tab is the second one, so the main tab survives.
		r = await repl.call(
			'exec',
			f"open_tab('{httpd}/a')\nopen_tab('{httpd}/b')\ncrashed = tabs()[-1]['targetId']\n"
			"try:\n\tcdp('Page.crash', _response_timeout=3)\nexcept Exception:\n\tpass\n"
			"import time; time.sleep(1.5)\ndrain_events()\nprint('crashed it')",
			30,
		)
		assert 'crashed it' in r['output'], r
		r = await repl.call(
			'exec',
			'try:\n\tswitch_to_tab(crashed)\nexcept Exception as e:\n\tprint(type(e).__name__, e)\n'
			"switch_to_tab(tabs()[0]['targetId'])\nprint('back on main', tab_stats()['crashes'])",
			30,
		)
		assert r['error'] is None, r['error']
		assert 'TabGoneError' in r['output'] and 'crashed' in r['output']
		assert 'back on main 1' in r['output']
	finally:
		await repl.close()


async def test_the_tab_limit_is_enforced_end_to_end(daemon, httpd, tmp_path):
	repl = Repl(tmp_path, {**daemon, 'BU_MAX_TABS': '2'})
	try:
		await repl.call('exec', RESET, 30)
		# The main tab is navigated first (not blank, so no reuse), the first open fills the last seat, the next is refused.
		r = await repl.call(
			'exec',
			f"goto_url('{httpd}/a')\nwait_for_load()\nimport time; time.sleep(0.5)\nopen_tab('{httpd}/b')\n"
			f"try:\n\topen_tab('{httpd}/c')\nexcept Exception as e:\n\tprint(type(e).__name__)\n"
			"print(tab_stats()['tab_limit_hits'], current_tab()['url'][:4])",
			30,
		)
		assert r['error'] is None, r['error']
		assert 'TabLimitError' in r['output']
		assert '1 http' in r['output']  # still on the tab it had, not stranded
	finally:
		await repl.close()


async def test_close_extra_tabs_leaves_only_the_main_tab(daemon, httpd, tmp_path):
	repl = Repl(tmp_path, daemon)
	try:
		await repl.call('exec', RESET, 30)
		r = await repl.call(
			'exec',
			f"goto_url('{httpd}/a')\nwait_for_load()\nimport time; time.sleep(0.5)\nopen_tab('{httpd}/b')\nopen_tab('{httpd}/c')\n"
			"print('closed', close_extra_tabs())\nprint([t['kind'] for t in tabs()], tab_stats()['tabs_open'])",
			30,
		)
		assert r['error'] is None, r['error']
		assert 'closed 2' in r['output'] and "['main'] 1" in r['output']
	finally:
		await repl.close()


async def test_await_popup_answers_none_when_the_popup_was_blocked(daemon, tmp_path):
	repl = Repl(tmp_path, daemon)
	try:
		await repl.call('exec', RESET, 30)
		r = await repl.call('exec', 'print(await_popup(0.5))', 30)
		assert r['error'] is None, r['error']
		assert 'None' in r['output']  # no popup came: an answer within the timeout, never a hang
	finally:
		await repl.close()
