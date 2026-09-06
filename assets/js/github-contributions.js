(() => {
  const script = document.currentScript;
  const username = script?.dataset.username;
  const calendar = document.querySelector('[data-contribution-calendar]');
  const months = document.querySelector('[data-contribution-months]');
  const summary = document.querySelector('[data-contribution-summary]');
  const fallback = document.querySelector('[data-contribution-fallback]');
  const scroll = document.querySelector('[data-contribution-scroll]');

  if (!username || !calendar || !summary) return;

  const formatDate = new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  const render = (items) => {
    const contributions = new Map(items.map((item) => [item.date, item]));
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const start = new Date(today);
    start.setMonth(start.getMonth() - 6);
    const gridStart = new Date(start);
    gridStart.setDate(gridStart.getDate() - gridStart.getDay());

    const fragment = document.createDocumentFragment();
    const monthStarts = [];
    let total = 0;
    let date = new Date(gridStart);

    while (date <= today) {
      const dayOffset = Math.round((date - gridStart) / 86400000);
      if (date.getDate() === 1 || dayOffset === 0) {
        monthStarts.push({
          label: `${date.getMonth() + 1}月`,
          week: Math.floor(dayOffset / 7)
        });
      }
      const key = [
        date.getFullYear(),
        String(date.getMonth() + 1).padStart(2, '0'),
        String(date.getDate()).padStart(2, '0')
      ].join('-');
      const item = contributions.get(key);
      const count = item?.count || 0;
      const level = Math.max(0, Math.min(4, item?.level || 0));
      const cell = document.createElement('span');
      const inRange = date >= start;

      if (inRange) total += count;
      cell.className = 'github-contribution-cell';
      cell.dataset.level = inRange ? String(level) : 'empty';
      cell.title = inRange
        ? `${formatDate.format(date)}：${count} 次贡献`
        : '';
      cell.setAttribute('aria-hidden', 'true');
      fragment.appendChild(cell);
      date.setDate(date.getDate() + 1);
    }

    calendar.replaceChildren(fragment);
    if (months) {
      const monthFragment = document.createDocumentFragment();
      monthStarts
        .filter((month, index) => {
          const next = monthStarts[index + 1];
          return !next || next.week - month.week >= 3;
        })
        .forEach((month) => {
          const label = document.createElement('span');
          label.textContent = month.label;
          label.style.setProperty('--month-week', String(month.week));
          monthFragment.appendChild(label);
        });
      months.replaceChildren(monthFragment);
    }
    summary.textContent = `过去六个月共 ${total.toLocaleString('zh-CN')} 次贡献`;
    calendar.setAttribute('aria-label', `${username} 过去六个月共 ${total} 次 GitHub 贡献`);
    scroll?.scrollTo({ left: scroll.scrollWidth });
  };

  const cacheKey = `gz-github-contributions-v1:${username}`;
  const freshFor = 60 * 60 * 1000;
  const usableFor = 7 * 24 * freshFor;
  const validItems = (items) => Array.isArray(items) && items.length > 0 && items.every((item) => (
    item && /^\d{4}-\d{2}-\d{2}$/.test(item.date) &&
    Number.isInteger(item.count) && item.count >= 0 &&
    Number.isInteger(item.level) && item.level >= 0 && item.level <= 4
  ));
  let cached = null;
  try {
    const value = JSON.parse(localStorage.getItem(cacheKey));
    const age = Date.now() - value?.savedAt;
    if (Number.isFinite(age) && age >= 0 && age < usableFor && validItems(value?.items)) cached = value;
  } catch {
    // Storage is optional; private browsing and corrupt entries must not break the calendar.
  }
  if (cached) {
    render(cached.items);
    if (Date.now() - cached.savedAt < freshFor) return;
    summary.textContent += '（上次记录，正在更新）';
  }

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 8000);

  fetch(`https://github-contributions-api.jogruber.de/v4/${encodeURIComponent(username)}?y=last`, {
    headers: { Accept: 'application/json' },
    signal: controller.signal
  })
    .then((response) => {
      if (!response.ok) throw new Error(`GitHub contributions request failed: ${response.status}`);
      return response.json();
    })
    .then((data) => {
      if (!validItems(data.contributions)) throw new Error('Unexpected contributions response');
      render(data.contributions);
      try {
        localStorage.setItem(cacheKey, JSON.stringify({ savedAt: Date.now(), items: data.contributions }));
      } catch {
        // A successful response remains usable when storage is unavailable.
      }
    })
    .catch(() => {
      if (cached) {
        render(cached.items);
        summary.textContent += '（上次记录，暂未更新）';
        return;
      }
      summary.textContent = '贡献记录暂时不可用';
      calendar.hidden = true;
      if (fallback) fallback.hidden = false;
    })
    .finally(() => window.clearTimeout(timeout));
})();
