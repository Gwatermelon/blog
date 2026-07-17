(() => {
  const initReadingToc = () => {
    const article = document.querySelector('.post-single');
    const content = article?.querySelector('.post-content');
    const toc = article?.querySelector(':scope > .toc');

    if (!article || !content || !toc) return;

    const links = [...toc.querySelectorAll('a[href^="#"]')];
    const headingLinks = links.map((link) => {
      const hash = link.getAttribute('href').slice(1);
      let id = hash;

      try {
        id = decodeURIComponent(hash);
      } catch (_) {
        // Keep the original hash when it is not URI encoded.
      }

      return { link, heading: document.getElementById(id), id };
    }).filter(({ heading }) => heading);

    if (!headingLinks.length) return;

    const progress = document.createElement('div');
    progress.className = 'reading-progress';
    progress.setAttribute('aria-hidden', 'true');
    progress.innerHTML = '<span></span>';
    document.body.append(progress);

    const progressBar = progress.firstElementChild;
    const tocInner = toc.querySelector('.inner');
    const desktopQuery = window.matchMedia('(min-width: 1360px)');
    let activeId = '';
    let frameRequested = false;

    const syncTocMode = () => {
      if (desktopQuery.matches) toc.open = true;
    };

    const revealActiveLink = (link) => {
      if (!desktopQuery.matches || !tocInner) return;

      const linkRect = link.getBoundingClientRect();
      const innerRect = tocInner.getBoundingClientRect();

      if (linkRect.top < innerRect.top + 6) {
        tocInner.scrollTop -= innerRect.top + 6 - linkRect.top;
      } else if (linkRect.bottom > innerRect.bottom - 6) {
        tocInner.scrollTop += linkRect.bottom - innerRect.bottom + 6;
      }
    };

    const setActive = ({ id, link }) => {
      if (id === activeId) return;
      activeId = id;

      links.forEach((tocLink) => {
        tocLink.classList.remove('is-active');
        tocLink.removeAttribute('aria-current');
      });
      toc.querySelectorAll('li.is-current-path').forEach((item) => {
        item.classList.remove('is-current-path');
      });

      link.classList.add('is-active');
      link.setAttribute('aria-current', 'location');

      let parent = link.closest('li');
      while (parent && toc.contains(parent)) {
        parent.classList.add('is-current-path');
        parent = parent.parentElement?.closest('li');
      }

      revealActiveLink(link);
    };

    const update = () => {
      frameRequested = false;
      const scrollY = window.scrollY;
      const markerY = scrollY + Math.min(180, window.innerHeight * .25);
      let current = headingLinks[0];

      for (const item of headingLinks) {
        if (item.heading.getBoundingClientRect().top + scrollY <= markerY) {
          current = item;
        } else {
          break;
        }
      }

      setActive(current);

      const contentTop = content.getBoundingClientRect().top + scrollY;
      const contentBottom = contentTop + content.offsetHeight;
      const readableDistance = Math.max(1, contentBottom - window.innerHeight - contentTop);
      const ratio = Math.min(1, Math.max(0, (scrollY - contentTop) / readableDistance));
      progressBar.style.transform = `scaleX(${ratio})`;
    };

    const requestUpdate = () => {
      if (frameRequested) return;
      frameRequested = true;
      window.requestAnimationFrame(update);
    };

    syncTocMode();
    update();
    window.addEventListener('scroll', requestUpdate, { passive: true });
    window.addEventListener('resize', requestUpdate, { passive: true });
    desktopQuery.addEventListener('change', () => {
      syncTocMode();
      requestUpdate();
    });

    if ('ResizeObserver' in window) {
      new ResizeObserver(requestUpdate).observe(content);
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initReadingToc, { once: true });
  } else {
    initReadingToc();
  }
})();
