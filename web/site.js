// Populate the shared navigation and project pages from the same editorial file.
async function renderSite() {
  const response = await fetch('content.json', { cache: 'no-store' });
  if (!response.ok) throw new Error('Could not read content.json');
  const { project, overview, dataViewer } = await response.json();
  const nav = document.querySelector('.site-nav');
  const page = location.pathname.split('/').pop() || 'index.html';

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function link({ label, href }, className) {
    const node = element('a', label, className);
    node.href = href;
    return node;
  }

  nav.setAttribute('aria-label', project.navigationLabel);
  nav.append(link({ label: project.brand, href: project.links[0].href }, 'site-brand'));
  const links = element('div', undefined, 'site-links');
  for (const item of project.links) {
    const node = link(item);
    if (item.href === page) node.setAttribute('aria-current', 'page');
    links.append(node);
  }
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
      legend.append(element('span', undefined, map.symbol), map.legend);
      const description = element('p', map.description);
      if (map.full) description.append(' ', link({ href: map.full, label: map.linkLabel }));
      caption.append(legend, element('h2', map.heading), description);
      figure.append(caption);
      maps.append(figure);
    }

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
