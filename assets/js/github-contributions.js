(() => {
  const script = document.currentScript;
  const username = script?.dataset.username;
  const calendar = document.querySelector('[data-contribution-calendar]');
  const summary = document.querySelector('[data-contribution-summary]');
  const fallback = document.querySelector('[data-contribution-fallback]');
  const scroll = document.querySelector('[data-contribution-scroll]');

  if (!username || !calendar || !summary) return;

  const formatDate = new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  const parseDate = (value) => {
    const [year, month, day] = value.split('-').map(Number);
    return new Date(year, month - 1, day);
  };

  const render = (items) => {
    const contributions = new Map(items.map((item) => [item.date, item]));
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const start = new Date(today);
    start.setDate(start.getDate() - 364);
    const gridStart = new Date(start);
    gridStart.setDate(gridStart.getDate() - gridStart.getDay());

    const fragment = document.createDocumentFragment();
    let total = 0;
    let date = new Date(gridStart);

    while (date <= today) {
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
    summary.textContent = `过去一年共 ${total.toLocaleString('zh-CN')} 次贡献`;
    calendar.setAttribute('aria-label', `${username} 过去一年共 ${total} 次 GitHub 贡献`);
    scroll?.scrollTo({ left: scroll.scrollWidth });
  };

  fetch(`https://github-contributions-api.jogruber.de/v4/${encodeURIComponent(username)}?y=last`, {
    headers: { Accept: 'application/json' }
  })
    .then((response) => {
      if (!response.ok) throw new Error(`GitHub contributions request failed: ${response.status}`);
      return response.json();
    })
    .then((data) => {
      if (!Array.isArray(data.contributions)) throw new Error('Unexpected contributions response');
      render(data.contributions);
    })
    .catch(() => {
      summary.textContent = '贡献记录暂时不可用';
      calendar.hidden = true;
      if (fallback) fallback.hidden = false;
    });
})();
