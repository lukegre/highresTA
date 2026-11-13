// Populate the shared navigation and project pages from the same editorial file.
async function renderSite() {
  const response = await fetch('content.json', { cache: 'no-store' });
  if (!response.ok) throw new Error('Could not read content.json');
  const { project, overview, dataViewer } = await response.json();
  const nav = document.querySelector('.site-nav');
  const page = location.pathname.split('/').pop() || 'index.html';
  if (page === 'explore.html' && !location.search) return location.replace(project.exploreHref);

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function link({ label, href }, className) {
    const node = element('a', label, className);
    // One saved explorer view for every link to the data explorer.
    node.href = href === 'explore.html' ? project.exploreHref : href;
    return node;
  }

  nav.setAttribute('aria-label', project.navigationLabel);
  nav.append(link({ label: project.brand, href: project.links[0].href }, 'site-brand'));
  const links = element('div', undefined, 'site-links');
  for (const item of project.links) {
    const node = link(item);
    if (item.href.split('?')[0] === page) node.setAttribute('aria-current', 'page');
    links.append(node);
  }
  const repo = link(project.repository, 'site-repo');
  repo.setAttribute('aria-label', project.repository.label);
  repo.title = project.repository.label;
  repo.target = '_blank';
  repo.rel = 'noopener';
  repo.innerHTML = '<svg viewBox="0 0 16 16" width="22" height="22" aria-hidden="true" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>';
  links.append(repo);
  nav.append(links);

  if (page === 'index.html') {
    document.title = overview.title;
    document.querySelector('meta[name="description"]').content = overview.intro;
    const subnav = document.querySelector('.section-nav');
    subnav.setAttribute('aria-label', overview.sectionNavLabel);
    for (const item of overview.sectionNav) subnav.append(link(item));
    function highlightSection() {
      const active = location.hash || overview.sectionNav[0]?.href;
      for (const item of subnav.querySelectorAll('a')) {
        if (item.hash === active) item.setAttribute('aria-current', 'location');
        else item.removeAttribute('aria-current');
      }
    }
    highlightSection();
    window.addEventListener('hashchange', highlightSection);
    const goal = document.getElementById('goal');
    goal.append(element('h1', overview.heading), element('p', overview.intro, 'lead'));
    const facts = element('dl', undefined, 'facts');
    for (const item of overview.facts) {
      const pair = element('div');
      pair.append(element('dt', item.value), element('dd', item.label));
      facts.append(pair);
    }
    goal.append(facts);

    const maps = document.getElementById('maps');
    maps.setAttribute('aria-label', overview.mapsLabel);
    for (const map of overview.maps) {
      const figure = element('figure');
      const image = element('img', undefined, 'atlas');
      image.src = map.image;
      image.alt = map.alt;
      figure.append(image);
      const caption = element('figcaption');
      const legend = element('p', undefined, 'map-meta');
      for (const key of map.legend) legend.append(element('span', undefined, key.symbol), key.label);
      const description = element('p', map.description);
      if (map.full) description.append(' ', link({ href: map.full, label: map.linkLabel }));
      caption.append(legend, element('h2', map.heading), description);
      figure.append(caption);
      maps.append(figure);
    }

    const model = document.getElementById('approach');
    model.append(element('h2', overview.model.heading), element('p', overview.model.intro, 'lead'),
      element('p', overview.model.equation, 'equation'));
    const parts = element('div', undefined, 'model-parts');
    for (const item of overview.model.items) {
      const part = element('article');
      part.append(element('h3', item.title), element('p', item.text));
      parts.append(part);
    }
    model.append(parts);

    const routes = document.getElementById('explore-project');
    routes.setAttribute('aria-label', overview.routesLabel);
    for (const item of overview.routes) {
      const article = element('article');
      const heading = element('h2');
      heading.append(link(item));
      article.append(heading, element('p', item.description));
      routes.append(article);
    }
    document.querySelector('footer').append(element('span', project.brand), link(project.repository));
  } else if (page === 'explore.html') {
    document.title = dataViewer.title;
    document.getElementById('viewer-error').textContent = dataViewer.error;
    document.getElementById('gv-app').setAttribute('aria-label', dataViewer.viewerTitle);
  }
}
renderSite().catch(error => {
  const status = document.createElement('p');
  status.setAttribute('role', 'alert');
  status.textContent = 'Could not load the page content. Serve the web folder over HTTP and reload.';
  document.body.prepend(status);
  console.error(error);
});
