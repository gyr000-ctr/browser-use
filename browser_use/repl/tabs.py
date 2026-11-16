"""Multi-tab/popup orchestration for the REPL worker (one TabManager per worker).

The default single-tab behavior is unchanged: the browser-harness helpers (new_tab, switch_tab, close_tab, list_tabs)
work as before. On top of them the worker gets open_tab(), switch_to_tab(), await_popup(), popups(), tabs(),
close_extra_tabs() and tab_stats(): discover (Target.* events, never a per-step Target.getTargets), register, switch,
wait for the load and backfill the page info. Bounded: MAX_TABS live tabs, MAX_POPUPS_PER_PAGE per opener and
MAX_POPUPS_TOTAL live popups overall (an infinite-popup page's further popups are closed as they arrive), MAX_REGISTRY
table entries. A crashed tab or a failed switch puts the agent back on the main tab instead of hanging. tab_stats()
reports tabs, crashes, downloads, popup-limit hits, fallbacks and worker memory, and every event is appended to
tabs.jsonl for audit.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

MAX_TABS = 8  # live tabs at once, the main tab included (BU_MAX_TABS)
MAX_POPUPS_PER_PAGE = 4  # live popups of one opener (BU_MAX_POPUPS_PER_PAGE)
MAX_POPUPS_TOTAL = 16  # live popups overall (BU_MAX_POPUPS_TOTAL): an infinite-popup page's are closed on arrival
MAX_REGISTRY = 64  # table entries, closed tabs included: a tab storm never grows it without bound
POPUP_WAIT_S = 5.0  # await_popup()'s default: a blocked window.open answers None within this, never hangs
LOAD_WAIT_S = 15.0  # how long open/switch wait for the load before backfilling what is there
MAX_NOTICES = 32  # unreported notices kept for the cell's end
MAX_GUIDS = 1024  # completed-download guids remembered for dedup

CRASH_EVENTS = ('Target.targetCrashed', 'Inspector.targetCrashed')
PROGRESS_EVENTS = ('Browser.downloadProgress', 'Page.downloadProgress')


class TabLimitError(RuntimeError):
	"""open_tab() at the live-tab limit: close a tab first."""


class TabGoneError(RuntimeError):
	"""A tab the registry marks closed or crashed, or never knew."""


def _memory_kb() -> int | None:
	"""Peak RSS of this (the worker's) process in KiB; None where resource is missing (Windows)."""
	try:
		import resource

		rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	except (ImportError, ValueError):
		return None
	return rss // 1024 if sys.platform == 'darwin' else rss  # macOS reports bytes, Linux KiB


class TabManager:
	"""The bounded tab/handle table and the orchestration on top of it.

	own_browser: this browser is the agent's own (cloud, or one Browser launched): every discovered tab is its to
	manage. False on the user's Chrome (Browser('chrome')): tabs without an opener are the user's and are registered
	(kind 'external') but never closed by cleanup(). on_event(name): telemetry counter (CellStats.count('events', ...)).
	"""

	def __init__(
		self,
		own_browser: bool = True,
		on_event: Callable[[str], None] | None = None,
		max_tabs: int = MAX_TABS,
		max_popups_per_page: int = MAX_POPUPS_PER_PAGE,
		max_popups_total: int = MAX_POPUPS_TOTAL,
		max_registry: int = MAX_REGISTRY,
		popup_wait_s: float = POPUP_WAIT_S,
		load_wait_s: float = LOAD_WAIT_S,
		log_path: Path | None = None,
		clock: Callable[[], float] = time.monotonic,
		sleep: Callable[[float], None] = time.sleep,
	):
		self.own_browser = own_browser
		self.on_event = on_event or (lambda name: None)
		self.max_tabs, self.max_popups_per_page, self.max_popups_total = max_tabs, max_popups_per_page, max_popups_total
		self.max_registry, self.popup_wait_s, self.load_wait_s = max_registry, popup_wait_s, load_wait_s
		self.log_path, self.clock, self.sleep = log_path, clock, sleep
		# one switch/open/close at a time: a background reader never races a cell's switch. Reentrant: a drain_events
		# triggered by the manager's own helper calls feeds observe() on the same thread.
		self.lock = threading.RLock()
		self.tabs: dict[str, dict] = {}  # target id -> record; bounded by max_registry
		self.main_id: str | None = None  # the tab failures fall back to
		self.attached: str | None = None  # the tab the daemon is attached to, as far as this manager knows
		self.cdp: Callable[..., Any] | None = None
		self.helpers: Any = None
		self.notices: list[str] = []
		# metrics (tab_stats(), the cell's stats['tabs'] and the tabs.jsonl audit records)
		self.opened = self.closed = self.popup_count = self.rejected = 0
		self.crashes = self.fallbacks = self.limit_hits = self.untracked = 0
		self.downloads_completed = 0
		self.peak = 0
		self._guids: set[str] = set()  # completed downloads already counted
		self._seq = 0  # popup registration order
		self._awaited = 0  # the popup sequence await_popup() already returned

	def bind(self, cdp: Callable[..., Any], helpers: Any) -> None:
		"""Give the manager the worker's timed cdp() and the helpers module (late: they are wrapped after __init__)."""
		self.cdp, self.helpers = cdp, helpers

	# --- registry ---

	def _register(
		self, tid: str, url: str = '', title: str = '', opener: str | None = None, kind: str = 'tab', own: bool = True
	) -> dict | None:
		if tid in self.tabs:
			tab = self.tabs[tid]
			tab['url'], tab['title'] = url or tab['url'], title or tab['title']
			return tab
		if len(self.tabs) >= self.max_registry:
			self.untracked += 1
			return None
		tab = {
			'id': tid,
			'url': url,
			'title': title,
			'opener': opener,
			'kind': kind,
			'own': own,
			'crashed': False,
			'closed': False,
		}
		self.tabs[tid] = tab
		self.peak = max(self.peak, self._live())
		return tab

	def _live(self) -> int:
		return sum(not t['closed'] for t in self.tabs.values())

	def _live_own(self) -> int:
		"""Live tabs the agent manages: the open() limit. Tabs of the user's own Chrome never count against it."""
		return sum(not t['closed'] and t['own'] for t in self.tabs.values())

	def _live_popups(self, opener: str | None = None) -> int:
		return sum(
			not t['closed'] and t['kind'] == 'popup' and (opener is None or t['opener'] == opener) for t in self.tabs.values()
		)

	def seed(self) -> None:
		"""One bounded Target list at worker start: the attached tab becomes the main tab. Never called per step."""
		current = self.helpers.current_tab()
		with self.lock:
			self._register(str(current['targetId']), str(current.get('url') or ''), str(current.get('title') or ''), kind='main')
			self.main_id = self.attached = str(current['targetId'])
			for t in self.helpers.list_tabs():
				if str(t['targetId']) != self.main_id:
					self._register(str(t['targetId']), str(t.get('url') or ''), str(t.get('title') or ''), own=self.own_browser)

	# --- event-driven discovery (drained Target.* events; no per-step enumeration) ---

	def observe(self, events: Iterable[dict] | None) -> None:
		close: list[str] = []
		for e in events or []:
			if not isinstance(e, dict):
				continue
			method, params = e.get('method'), e.get('params') or {}
			if method == 'Target.targetCreated':
				info = params.get('targetInfo') or {}
				if info.get('type') == 'page':
					close += self._discovered(info)
			elif method == 'Target.targetInfoChanged':
				info = params.get('targetInfo') or {}
				with self.lock:
					if (tab := self.tabs.get(str(info.get('targetId') or ''))) is not None:
						tab['url'], tab['title'] = str(info.get('url') or tab['url']), str(info.get('title') or tab['title'])
			elif method == 'Target.targetDestroyed':
				self._destroyed(str(params.get('targetId') or ''))
			elif method in CRASH_EVENTS:
				self._crashed()
			elif method in PROGRESS_EVENTS and params.get('state') == 'completed':
				guid = str(params.get('guid') or '')
				with self.lock:
					if guid not in self._guids:
						if len(self._guids) < MAX_GUIDS:
							self._guids.add(guid)
						self.downloads_completed += 1
						self._count('download_completed')
		for tid in close:  # outside the registry lock: a CDP call of its own
			with contextlib.suppress(Exception):  # already gone; unbound (tests): nothing to close with
				if self.cdp is not None:
					self.cdp('Target.closeTarget', targetId=tid)

	def _discovered(self, info: dict) -> list[str]:
		"""Register a page target Chrome announced. Returns ids to close (popups over the budget)."""
		tid = str(info.get('targetId') or '')
		if not tid:
			return []
		opener = info.get('openerId') or None
		url = str(info.get('url') or '')
		with self.lock:
			if tid in self.tabs:
				return []
			if opener:  # a window.open()/target=_blank popup of a page
				if self._live_popups() >= self.max_popups_total or self._live_popups(opener) >= self.max_popups_per_page:
					self.rejected += 1
					self._count('popup_rejected')
					self._notice(f'[popup limit reached: closed {url or tid}]')
					self._log({'event': 'popup_rejected', 'tab': tid, 'url': url, 'opener': opener})
					return [tid]
				kind, own = 'popup', True
			else:
				kind, own = 'tab', self.own_browser
			tab = self._register(tid, url, str(info.get('title') or ''), opener, kind, own)
			if tab is None:
				return []
			if kind == 'popup':
				self.popup_count += 1
				self._seq += 1
				tab['seq'] = self._seq
				self._count('popup_opened')
				self._notice(f"[popup opened: {url or tid} (targetId {tid}); switch_to_tab('{tid}') to work in it]")
				self._log({'event': 'popup', 'tab': tid, 'url': url, 'opener': opener})
		return []

	def _destroyed(self, tid: str) -> None:
		with self.lock:
			if (tab := self.tabs.get(tid)) is not None and not tab['closed']:
				tab['closed'] = True
				self.closed += 1
				self._log({'event': 'closed', 'tab': tid, 'url': tab['url']})

	def _crashed(self) -> None:
		"""A Target/Inspector.targetCrashed reaches this daemon for the tab it is attached to."""
		with self.lock:
			tab = self.tabs.get(self.attached or '')
			if tab is None or tab['crashed']:
				return
			tab['crashed'] = True
			self.crashes += 1
			self._count('tab_crashed')
			self._notice(
				f'[tab crashed: {tab["url"] or tab["id"]}; the main tab is still there: switch back with switch_to_tab()]'
			)
			self._log({'event': 'crashed', 'tab': tab['id'], 'url': tab['url']})

	# --- orchestration ---

	def open(self, url: str = 'about:blank') -> dict:
		"""Open a tab, wait for the load and return its page_info. Past MAX_TABS live tabs raises TabLimitError; any
		failure leaves the agent back on the main tab."""
		with self.lock:
			if self._live_own() >= self.max_tabs:
				self.limit_hits += 1
				self._count('tab_limit')
				self._log({'event': 'tab_limit', 'live': self._live_own()})
				raise TabLimitError(
					f'the tab limit ({self.max_tabs}) is reached; close one with close_tab() or close_extra_tabs() first'
				)
			try:
				tid = str(self.helpers.new_tab(url))
				self.attached = tid
				tab = self._register(tid, kind='tab', own=True)
				if tab is not None and tab['kind'] != 'main':
					self.opened += 1
					self._count('tab_opened')
					self._log({'event': 'opened', 'tab': tid, 'url': url})
				self.helpers.wait_for_load(self.load_wait_s)
				info = self.helpers.page_info()
			except Exception:
				self._fallback(f'open_tab({url!r}) failed')
				raise
			if tab is not None and isinstance(info, dict):
				self._update(tab, info)
			return info

	def switch(self, target_id: str) -> dict:
		"""Attach a registered tab, wait for its load and return its page_info. A crashed/closed tab raises
		TabGoneError; any other failure leaves the agent back on the main tab."""
		tid = str(target_id)
		with self.lock:
			tab = self.tabs.get(tid)
			if tab is None:
				raise TabGoneError(f'unknown tab {tid}; tabs() lists the registered ones')
			if tab['closed'] or tab['crashed']:
				raise TabGoneError(f'tab {tid} is {"crashed" if tab["crashed"] else "closed"} ({tab["url"] or "no URL"})')
			try:
				self.helpers.switch_tab(tab['id'])
				self.attached = tid
				self.helpers.wait_for_load(self.load_wait_s)
				info = self.helpers.page_info()
			except Exception:
				self._fallback(f'could not switch to tab {tid}')
				raise
			if isinstance(info, dict):
				self._update(tab, info)
			return info

	def await_popup(self, timeout: float | None = None) -> dict | None:
		"""The first popup registered since the last await_popup(), or None when none showed within timeout (a
		blocked window.open() ends here, never in a hang)."""
		deadline = self.clock() + (self.popup_wait_s if timeout is None else timeout)
		while True:
			with self.lock:
				new = [
					t for t in self.tabs.values() if t['kind'] == 'popup' and not t['closed'] and t.get('seq', 0) > self._awaited
				]
				if new:
					new.sort(key=lambda t: t['seq'])
					self._awaited = new[0]['seq']
					return self._public(new[0])
			if self.clock() >= deadline:
				return None
			with contextlib.suppress(Exception):  # a daemon hiccup: the deadline still bounds the wait
				self.helpers.drain_events()
			self.sleep(0.1)

	def popups(self) -> list[dict]:
		"""The live popups of this run's pages."""
		with self.lock:
			return [self._public(t) for t in self.tabs.values() if t['kind'] == 'popup' and not t['closed']]

	def listing(self) -> list[dict]:
		"""The registered live tabs (bounded), main first. Never a fresh Target.getTargets."""
		with self.lock:
			live = [t for t in self.tabs.values() if not t['closed']]
			live.sort(key=lambda t: t['id'] != self.main_id)
			return [self._public(t) for t in live]

	def cleanup(self) -> int:
		"""Close every own tab but the main one (never an external tab of the user's Chrome). Returns how many it closed."""
		with self.lock:
			closed = 0
			for tab in list(self.tabs.values()):
				if tab['id'] == self.main_id or tab['closed'] or not tab['own']:
					continue
				self._close(tab)
				closed += tab['closed']
			if self.attached != self.main_id and self.main_id and self.tabs.get(self.attached or '', {}).get('closed', True):
				with contextlib.suppress(Exception):  # the daemon attaches another tab itself
					self.helpers.switch_tab(self.main_id)
					self.attached = self.main_id
			return closed

	# --- reporting ---

	def metrics(self) -> dict:
		"""tab_stats(): tab, crash, download, popup-limit, fallback and memory numbers for tuning and audit."""
		with self.lock:
			return {
				'tabs_open': self._live(),
				'tabs_opened': self.opened,
				'tabs_closed': self.closed,
				'popups': self.popup_count,
				'popups_rejected': self.rejected,
				'crashes': self.crashes,
				'fallbacks': self.fallbacks,
				'downloads_completed': self.downloads_completed,
				'tab_limit_hits': self.limit_hits,
				'untracked': self.untracked,
				'peak_tabs': self.peak,
				'memory_kb': _memory_kb(),
			}

	def take_notices(self) -> list[str]:
		with self.lock:
			out, self.notices = self.notices, []
			return out

	# --- internals ---

	def _update(self, tab: dict, info: dict) -> None:
		tab['url'] = str(info.get('url') or tab['url'])
		tab['title'] = str(info.get('title') or tab['title'])

	def _public(self, tab: dict) -> dict:
		return {
			'targetId': tab['id'],
			'url': tab['url'],
			'title': tab['title'],
			'kind': tab['kind'],
			'opener': tab['opener'],
			'crashed': tab['crashed'],
		}

	def _close(self, tab: dict) -> None:
		with contextlib.suppress(Exception):  # already gone: the destroyed event marks it
			if self.cdp is not None:
				self.cdp('Target.closeTarget', targetId=tab['id'])
		if not tab['closed']:
			tab['closed'] = True
			self.closed += 1
			self._log({'event': 'closed', 'tab': tab['id'], 'url': tab['url']})

	def _fallback(self, why: str) -> None:
		"""Back to the main tab after a failed open/switch: the agent is never left attached to a broken tab."""
		self.fallbacks += 1
		self._count('tab_fallback')
		self._notice(f'[tab: {why}; back on the main tab]')
		self._log({'event': 'fallback', 'why': why})
		with contextlib.suppress(Exception):  # the main tab or browser is gone too: the reattach flow reports that
			main = self.tabs.get(self.main_id or '')
			if main and not main['closed']:
				self.helpers.switch_tab(main['id'])
				self.attached = main['id']

	def _count(self, name: str) -> None:
		with contextlib.suppress(Exception):  # telemetry never breaks a cell
			self.on_event(name)

	def _notice(self, line: str) -> None:
		if len(self.notices) < MAX_NOTICES:
			self.notices.append(line)

	def _log(self, record: dict) -> None:
		if self.log_path is None:
			return
		with contextlib.suppress(OSError), open(self.log_path, 'a', encoding='utf-8') as f:
			f.write(json.dumps({'at': round(time.time(), 3), **record}) + '\n')
