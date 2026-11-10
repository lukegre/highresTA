
// CDN imports (pinned, as in the reference preview.html). Must stay the first lines of the module script.
import Map from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/Map.js/+esm';
import WMTSTileGrid from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/tilegrid/WMTS.js/+esm';
import {getView, withExtentCenter, withHigherResolutions, withLowerResolutions, withZoom}
  from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/View.js/+esm';
import WebGLTileLayer from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/layer/WebGLTile.js/+esm';
import TileLayer from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/layer/Tile.js/+esm';
import GeoZarr from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/source/GeoZarr.js/+esm';
import OSM from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/source/OSM.js/+esm';
import XYZ from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/source/XYZ.js/+esm';
import VectorLayer from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/layer/Vector.js/+esm';
import VectorSource from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/source/Vector.js/+esm';
import TopoJSON from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/format/TopoJSON.js/+esm';
import {register} from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/proj/proj4.js/+esm';
import {toLonLat} from 'https://cdn.jsdelivr.net/npm/ol@10.10.0/proj.js/+esm';
import proj4 from 'https://cdn.jsdelivr.net/npm/proj4@2.22.0/+esm';
// Must be the zarrita version that ol@10.10.0's GeoZarr source imports, so this is the module instance OL uses.
// S3 CORS rules that allow only GET leave HEAD responses without CORS headers (D-56), see noHeadSuffixReads.
import {FetchStore as OlFetchStore} from 'https://cdn.jsdelivr.net/npm/zarrita@0.7.3/+esm';
// browserConfig reads metadata with its own (newer) zarrita, loaded on demand.
const ZARRITA_URL = 'https://cdn.jsdelivr.net/npm/zarrita@0.7.5/+esm';

// ---- core/util.js
// Shared helpers: DOM, formatting, icons, namespaced storage. Reference: the helpers around `h`, `fmt`, `ICON`, `store`.
const $ = (id) => document.getElementById(id);
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'})[c]);
const codeify = (t) => esc(t).replace(/`([^`]+)`/g, '<code>$1</code>');
const fmt = (x) => String(+Number(x).toPrecision(6));
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const css = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
const unitsOf = (v) => v.units ?? '';
const dimLabel = (d, i) => (d.labels && d.labels[i] !== undefined ? d.labels[i] : String(i));

// h('div', 'cls', 'text') -> element
function h(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
}

// kvTable([[key, value, id?], ...]) -> <table class="gv-kv">; `id` (optional) becomes data-id of the value cell.
function kvTable(rows) {
  const tbl = h('table', 'gv-kv'), body = h('tbody');
  for (const [k, v, id] of rows) {
    const tr = h('tr'), td = h('td', '', String(v));
    if (id) td.dataset.id = id;
    tr.append(h('th', '', k), td);
    body.append(tr);
  }
  tbl.append(body);
  return tbl;
}

// Keep the role of every dim that has one; a new dim is horizontal only if no dim holds that role (D-40).
// Idempotent, so the order timeline, dim-sliders and dim-roles run in does not matter.
function syncRoles(dimNames, roles) {
  let holder = dimNames.find((d) => roles[d] === 'h');
  for (const d of dimNames) {
    if (!roles[d]) roles[d] = holder === undefined ? 'h' : 'v';
    if (roles[d] === 'h') { if (holder === undefined) holder = d; else if (holder !== d) roles[d] = 'v'; }
  }
  return roles;
}

const hasModifier = (e) => e.altKey || e.ctrlKey || e.metaKey || e.shiftKey;
const isTyping = (e) => Boolean(e.target?.closest?.('input, select, textarea, [contenteditable]'));
// A keydown without Alt/Ctrl/Meta/Shift that is not aimed at a field (D-50): the arrow-key shortcuts use it.
const plainKey = (e) => !hasModifier(e) && !isTyping(e);

const ICON = {
  eye: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5z" fill="none" stroke="currentColor"/><circle cx="8" cy="8" r="2" fill="none" stroke="currentColor"/></svg>',
  eyeOff: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5zM2 2l12 12" fill="none" stroke="currentColor"/><circle cx="8" cy="8" r="2" fill="none" stroke="currentColor"/></svg>',
  play: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>',
  pause: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor"/></svg>',
  prev: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3 5 8l5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  next: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="m6 3 5 5-5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  chevDown: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  info: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M8 7.2v4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="8" cy="4.9" r=".9" fill="currentColor"/></svg>',
  gear: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 4h7M13 4h1M2 12h1M7 12h7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="11" cy="4" r="2" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="5" cy="12" r="2" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>',
};

// Namespaced localStorage (keys get the prefix) with an in-memory fallback when storage is unavailable
// (private windows, blocked site data, file:// in some browsers).
function makeStorage(prefix = 'gv:') {
  const mem = new globalThis.Map();  // `Map` is the OpenLayers class in the page module
  let ls;
  try { ls = localStorage; } catch (_) { ls = null; }
  const guard = (fn, fallback) => { try { return ls ? fn(ls) : fallback(); } catch (_) { return fallback(); } };
  return {
    get: (key) => guard((s) => s.getItem(prefix + key), () => (mem.has(prefix + key) ? mem.get(prefix + key) : null)),
    set: (key, value) => guard((s) => s.setItem(prefix + key, String(value)), () => { mem.set(prefix + key, String(value)); }),
    remove: (key) => guard((s) => s.removeItem(prefix + key), () => { mem.delete(prefix + key); }),
  };
}

// ---- core/colormaps.js
// Generated by scripts/extract_colormaps.py from the reference preview.html config; do not edit.
// COLORMAPS: name -> 11 RGB stops. COLORMAP_GROUPS: picker groups (source core = matplotlib, cmocean, cmasher).
const COLORMAPS = {"viridis":[[68,1,84],[72,36,117],[65,68,135],[53,95,141],[42,120,142],[33,145,140],[34,168,132],[68,191,112],[122,209,81],[189,223,38],[253,231,37]],"plasma":[[13,8,135],[65,4,157],[106,0,168],[143,13,164],[177,42,144],[204,71,120],[225,100,98],[242,132,75],[252,166,54],[252,206,37],[240,249,33]],"inferno":[[0,0,4],[22,11,57],[66,10,104],[106,23,110],[147,38,103],[188,55,84],[221,81,58],[243,120,25],[252,165,10],[246,215,70],[252,255,164]],"magma":[[0,0,4],[20,14,54],[59,15,112],[100,26,128],[140,41,129],[183,55,121],[222,73,104],[247,112,92],[254,159,109],[254,207,146],[252,253,191]],"cividis":[[0,34,78],[8,51,112],[53,69,108],[79,87,108],[102,105,112],[125,124,120],[148,142,119],[174,163,113],[200,184,102],[229,207,82],[254,232,56]],"Greys":[[255,255,255],[243,243,243],[226,226,226],[206,206,206],[181,181,181],[149,149,149],[122,122,122],[95,95,95],[64,64,64],[29,29,29],[0,0,0]],"Purples":[[252,251,253],[242,240,247],[226,226,239],[206,207,229],[182,182,216],[158,154,200],[134,131,189],[114,98,172],[97,64,155],[79,31,139],[63,0,125]],"Blues":[[247,251,255],[227,238,249],[208,225,242],[183,212,234],[148,196,223],[106,174,214],[74,152,201],[46,126,188],[23,100,171],[8,74,145],[8,48,107]],"Greens":[[247,252,245],[233,247,229],[211,238,205],[184,227,178],[152,213,148],[115,196,118],[75,176,98],[47,151,78],[21,127,59],[0,100,40],[0,68,27]],"Oranges":[[255,245,235],[254,233,212],[253,217,180],[253,195,141],[253,167,98],[253,140,59],[243,112,27],[226,85,8],[197,65,2],[158,51,3],[127,39,4]],"Reds":[[255,245,240],[254,229,216],[253,202,181],[252,171,143],[252,138,106],[251,105,74],[241,68,50],[217,37,35],[188,20,26],[152,12,19],[103,0,13]],"YlOrBr":[[255,255,229],[255,249,197],[254,235,162],[254,215,120],[254,187,71],[254,152,41],[240,120,24],[216,90,9],[184,66,3],[142,49,4],[102,37,6]],"YlOrRd":[[255,255,204],[255,241,169],[254,225,135],[254,202,102],[254,171,73],[253,140,60],[252,91,46],[237,46,33],[212,16,32],[176,0,38],[128,0,38]],"OrRd":[[255,247,236],[254,235,208],[253,220,175],[253,202,148],[253,178,123],[252,140,89],[242,109,75],[224,68,47],[201,29,19],[168,0,0],[127,0,0]],"PuRd":[[247,244,249],[234,229,241],[220,201,226],[208,171,211],[205,139,194],[223,100,175],[229,53,146],[216,27,106],[184,11,78],[141,0,59],[103,0,31]],"RdPu":[[255,247,243],[253,229,226],[252,208,204],[251,182,188],[249,148,177],[247,103,161],[226,62,153],[192,21,136],[153,1,123],[111,1,116],[73,0,106]],"BuPu":[[247,252,253],[229,239,246],[204,221,236],[178,202,225],[154,180,214],[140,149,198],[140,116,181],[138,81,165],[133,45,144],[118,12,113],[77,0,75]],"GnBu":[[247,252,240],[229,245,224],[212,238,206],[190,230,191],[159,218,184],[122,204,196],[87,184,208],[56,155,198],[29,126,183],[8,95,163],[8,64,129]],"PuBu":[[255,247,251],[240,234,244],[219,218,235],[192,201,226],[156,185,217],[115,169,207],[66,149,195],[24,124,182],[5,103,162],[4,83,130],[2,56,88]],"YlGnBu":[[255,255,217],[241,250,186],[214,239,179],[171,222,183],[115,200,189],[64,181,196],[36,152,193],[32,114,177],[35,77,160],[31,47,135],[8,29,88]],"PuBuGn":[[255,247,251],[240,231,242],[219,216,234],[192,201,226],[153,185,217],[102,169,207],[64,149,195],[22,135,159],[2,121,118],[1,100,81],[1,70,54]],"BuGn":[[247,252,253],[233,247,250],[214,240,238],[184,228,219],[143,212,194],[101,194,163],[72,178,127],[47,152,88],[21,127,59],[0,100,40],[0,68,27]],"YlGn":[[255,255,229],[249,253,194],[229,245,172],[200,233,155],[162,216,138],[119,198,121],[76,176,99],[47,147,77],[21,121,62],[0,96,52],[0,69,41]],"binary":[[255,255,255],[230,230,230],[204,204,204],[179,179,179],[153,153,153],[127,127,127],[102,102,102],[76,76,76],[51,51,51],[25,25,25],[0,0,0]],"gist_yarg":[[255,255,255],[230,230,230],[204,204,204],[179,179,179],[153,153,153],[127,127,127],[102,102,102],[76,76,76],[51,51,51],[25,25,25],[0,0,0]],"gist_gray":[[0,0,0],[25,25,25],[51,51,51],[76,76,76],[102,102,102],[128,128,128],[153,153,153],[179,179,179],[204,204,204],[230,230,230],[255,255,255]],"gray":[[0,0,0],[25,25,25],[51,51,51],[76,76,76],[102,102,102],[128,128,128],[153,153,153],[179,179,179],[204,204,204],[230,230,230],[255,255,255]],"bone":[[0,0,0],[22,22,30],[45,45,62],[66,66,93],[89,92,121],[112,123,144],[134,154,166],[157,185,188],[185,210,210],[221,233,233],[255,255,255]],"pink":[[30,0,0],[104,65,65],[145,93,93],[175,114,114],[198,139,132],[208,172,148],[218,198,161],[228,223,174],[237,237,198],[247,247,229],[255,255,255]],"spring":[[255,0,255],[255,25,230],[255,51,204],[255,76,179],[255,102,153],[255,128,127],[255,153,102],[255,179,76],[255,204,51],[255,230,25],[255,255,0]],"summer":[[0,128,102],[25,140,102],[51,153,102],[76,166,102],[102,178,102],[128,192,102],[153,204,102],[179,217,102],[204,230,102],[230,242,102],[255,255,102]],"autumn":[[255,0,0],[255,25,0],[255,51,0],[255,76,0],[255,102,0],[255,128,0],[255,153,0],[255,179,0],[255,204,0],[255,230,0],[255,255,0]],"winter":[[0,0,255],[0,25,242],[0,51,230],[0,76,217],[0,102,204],[0,128,191],[0,153,178],[0,179,166],[0,204,153],[0,230,140],[0,255,128]],"cool":[[0,255,255],[25,230,255],[51,204,255],[76,179,255],[102,153,255],[128,127,255],[153,102,255],[179,76,255],[204,51,255],[230,25,255],[255,0,255]],"Wistia":[[228,255,122],[239,246,84],[250,237,45],[255,224,21],[255,206,10],[255,189,0],[255,177,0],[255,166,0],[254,153,0],[253,140,0],[252,127,0]],"hot":[[11,0,0],[76,0,0],[144,0,0],[210,0,0],[255,23,0],[255,92,0],[255,157,0],[255,225,0],[255,255,54],[255,255,157],[255,255,255]],"afmhot":[[0,0,0],[50,0,0],[102,0,0],[152,24,0],[204,77,0],[255,128,1],[255,178,51],[255,230,103],[255,255,153],[255,255,205],[255,255,255]],"gist_heat":[[0,0,0],[38,0,0],[77,0,0],[114,0,0],[153,0,0],[192,1,0],[229,51,0],[255,103,0],[255,153,51],[255,205,155],[255,255,255]],"copper":[[0,0,0],[31,20,12],[63,40,25],[94,59,38],[126,80,51],[158,100,64],[189,120,76],[221,140,89],[252,159,101],[255,180,114],[255,199,127]],"PiYG":[[142,1,82],[196,26,124],[222,119,174],[241,181,217],[253,224,239],[247,247,246],[230,245,208],[183,224,133],[127,188,65],[76,145,33],[39,100,25]],"PRGn":[[64,0,75],[117,41,130],[153,112,171],[193,164,206],[231,212,232],[246,247,246],[217,240,211],[165,218,159],[90,174,97],[26,119,54],[0,68,27]],"BrBG":[[84,48,5],[139,80,10],[191,129,45],[222,193,123],[246,232,195],[244,245,245],[199,234,229],[127,204,192],[53,151,143],[1,101,93],[0,60,48]],"PuOr":[[127,59,8],[178,87,6],[224,130,20],[252,183,97],[254,224,182],[246,246,247],[216,218,235],[177,170,209],[128,115,172],[83,38,135],[45,0,75]],"RdGy":[[103,0,31],[177,24,43],[214,96,77],[243,164,129],[253,219,199],[254,254,254],[224,224,224],[185,185,185],[135,135,135],[76,76,76],[26,26,26]],"RdBu":[[103,0,31],[177,24,43],[214,96,77],[243,164,129],[253,219,199],[246,247,247],[209,229,240],[144,196,221],[67,147,195],[32,101,171],[5,48,97]],"RdBu_r":[[5,48,97],[32,101,171],[67,147,195],[144,196,221],[209,229,240],[247,246,246],[253,219,199],[243,164,129],[214,96,77],[177,24,43],[103,0,31]],"RdYlBu":[[165,0,38],[214,47,39],[244,109,67],[253,173,96],[254,224,144],[254,255,192],[224,243,248],[170,216,233],[116,173,209],[69,116,179],[49,54,149]],"RdYlGn":[[165,0,38],[214,47,39],[244,109,67],[253,173,96],[254,224,139],[254,255,190],[217,239,139],[165,216,106],[102,189,99],[25,151,80],[0,104,55]],"Spectral":[[158,1,66],[212,61,79],[244,109,67],[253,173,96],[254,224,139],[255,255,190],[230,245,152],[170,220,164],[102,194,165],[51,135,188],[94,79,162]],"coolwarm":[[59,76,192],[89,119,227],[123,159,249],[158,190,255],[192,212,245],[221,220,220],[242,203,183],[247,172,142],[238,132,104],[214,82,68],[180,4,38]],"bwr":[[0,0,255],[50,50,255],[102,102,255],[152,152,255],[204,204,255],[255,254,254],[255,204,204],[255,152,152],[255,102,102],[255,50,50],[255,0,0]],"seismic":[[0,0,76],[0,0,146],[0,0,219],[49,49,255],[153,153,255],[255,253,253],[255,153,153],[255,49,49],[230,0,0],[178,0,0],[128,0,0]],"berlin":[[158,176,255],[98,166,224],[50,128,166],[32,82,106],[17,39,50],[25,12,9],[55,16,0],[97,32,11],[150,74,54],[202,123,113],[255,173,173]],"managua":[[255,207,103],[224,159,87],[193,116,73],[160,81,62],[119,51,57],[87,41,73],[76,61,115],[81,99,162],[95,137,195],[111,182,226],[129,231,255]],"vanimo":[[255,205,253],[216,137,202],[173,83,154],[116,46,100],[51,23,44],[26,21,19],[35,44,20],[65,90,31],[98,135,47],[138,188,83],[190,253,165]],"twilight":[[226,217,226],[166,191,202],[109,144,192],[95,88,176],[83,30,124],[47,20,54],[100,25,75],[159,60,80],[192,117,94],[208,179,158],[226,217,226]],"twilight_shifted":[[48,20,55],[84,30,126],[95,89,177],[110,145,192],[167,192,203],[226,217,226],[208,178,156],[192,116,93],[158,59,80],[99,24,75],[47,20,54]],"hsv":[[255,0,0],[255,148,0],[209,255,0],[61,255,0],[0,255,92],[0,255,246],[0,116,255],[37,0,255],[185,0,255],[255,0,172],[255,0,24]],"flag":[[255,0,0],[0,0,221],[255,206,157],[0,0,47],[214,243,255],[164,0,0],[41,79,255],[255,88,47],[0,0,157],[255,245,221],[0,0,0]],"prism":[[255,0,0],[255,0,0],[255,42,0],[255,72,0],[255,149,0],[255,215,0],[255,235,0],[218,255,0],[189,255,0],[111,255,0],[84,255,0]],"ocean":[[0,128,0],[0,90,25],[0,51,51],[0,14,76],[0,26,102],[0,64,128],[0,102,153],[27,141,179],[102,179,204],[180,218,230],[255,255,255]],"gist_earth":[[0,0,0],[17,41,119],[34,94,124],[50,130,123],[62,145,91],[94,160,75],[142,172,86],[183,182,94],[196,164,111],[225,191,176],[253,251,251]],"terrain":[[51,51,153],[18,118,220],[0,178,178],[49,214,112],[153,235,133],[254,254,152],[204,190,125],[152,123,97],[153,124,118],[205,191,188],[255,255,255]],"gist_stern":[[0,0,0],[200,25,50],[70,51,102],[76,76,152],[102,102,204],[128,128,253],[153,153,146],[179,179,36],[204,204,63],[230,230,161],[255,255,255]],"gnuplot":[[0,0,0],[80,0,147],[114,2,243],[139,7,243],[161,16,150],[181,32,0],[198,55,0],[214,88,0],[228,131,0],[242,187,0],[255,255,0]],"gnuplot2":[[0,0,0],[0,0,100],[0,0,204],[38,0,255],[120,0,255],[201,42,213],[255,92,163],[255,144,111],[255,194,61],[255,246,9],[255,255,255]],"CMRmap":[[0,0,0],[30,30,100],[61,38,166],[106,43,167],[173,54,110],[255,65,38],[235,115,8],[230,167,16],[230,207,66],[235,235,155],[255,255,255]],"cubehelix":[[0,0,0],[26,20,47],[22,61,78],[30,101,66],[84,121,47],[161,121,74],[208,126,147],[207,157,219],[193,202,243],[211,238,239],[255,255,255]],"brg":[[0,0,255],[50,0,205],[102,0,153],[152,0,103],[204,0,51],[254,1,0],[204,51,0],[152,103,0],[102,153,0],[50,205,0],[0,255,0]],"gist_rainbow":[[255,0,41],[255,94,0],[255,234,0],[141,255,0],[0,255,0],[0,255,140],[0,236,255],[0,94,255],[42,0,255],[183,0,255],[255,0,191]],"rainbow":[[128,0,255],[78,77,252],[25,150,243],[24,205,228],[77,243,206],[128,255,180],[178,243,150],[230,205,115],[255,150,79],[255,77,39],[255,0,0]],"jet":[[0,0,128],[0,0,241],[0,76,255],[0,176,255],[41,255,206],[125,255,122],[206,255,41],[255,196,0],[255,104,0],[241,8,0],[128,0,0]],"turbo":[[48,18,59],[69,89,203],[62,155,254],[25,213,205],[70,248,132],[164,252,60],[225,221,55],[254,164,49],[240,91,18],[195,37,3],[122,4,3]],"nipy_spectral":[[0,0,0],[135,0,152],[0,0,221],[0,152,221],[0,170,136],[0,188,0],[0,255,0],[239,237,0],[255,153,0],[220,0,0],[204,204,204]],"gist_ncar":[[0,0,128],[0,14,205],[0,237,255],[0,253,63],[116,232,0],[219,255,32],[255,206,5],[255,52,0],[241,7,255],[236,132,239],[254,248,254]],"cmo.thermal":[[4,35,51],[15,50,106],[64,52,159],[103,67,150],[139,83,141],[177,95,130],[214,108,108],[243,130,77],[252,166,60],[247,208,69],[232,250,91]],"cmo.haline":[[42,24,108],[41,39,162],[13,78,150],[24,102,140],[45,124,137],[60,147,135],[74,170,129],[101,193,114],[148,211,93],[208,224,109],[253,239,154]],"cmo.solar":[[51,20,24],[82,30,34],[115,39,36],[143,52,31],[165,74,23],[182,100,19],[196,127,21],[207,156,29],[216,186,42],[222,219,58],[225,253,75]],"cmo.ice":[[4,6,19],[27,27,55],[48,47,95],[61,67,137],[62,94,169],[66,123,183],[82,150,193],[107,177,203],[140,203,214],[188,228,231],[234,253,253]],"cmo.gray":[[0,0,0],[19,18,18],[43,43,43],[66,66,65],[90,89,89],[114,113,113],[138,137,137],[164,164,163],[192,191,190],[222,222,221],[255,255,253]],"cmo.oxy":[[64,5,5],[106,6,15],[80,79,79],[103,102,102],[129,128,127],[155,154,154],[183,183,182],[213,213,212],[248,254,105],[231,215,44],[221,175,25]],"cmo.deep":[[253,254,204],[202,235,178],[146,216,164],[102,194,164],[82,168,163],[72,142,158],[64,117,152],[62,90,146],[65,64,123],[56,45,81],[40,26,44]],"cmo.dense":[[230,241,241],[188,220,229],[150,197,226],[123,173,228],[115,144,227],[120,113,213],[121,84,187],[116,58,152],[104,36,113],[83,21,70],[54,14,36]],"cmo.algae":[[215,249,208],[184,227,172],[150,205,138],[114,187,108],[68,168,85],[18,148,80],[9,125,75],[21,102,65],[26,80,52],[24,58,37],[18,36,20]],"cmo.matter":[[254,237,176],[250,202,144],[245,167,115],[238,133,94],[226,98,83],[206,67,86],[179,46,95],[147,32,99],[114,26,96],[79,22,82],[47,15,62]],"cmo.turbid":[[233,246,171],[219,217,135],[207,188,102],[196,161,77],[181,135,64],[161,112,59],[138,94,58],[112,77,55],[86,62,48],[59,47,39],[34,31,27]],"cmo.speed":[[255,253,205],[238,223,152],[216,197,95],[184,178,47],[142,162,11],[95,146,12],[50,128,31],[15,107,43],[16,84,44],[25,59,35],[23,35,19]],"cmo.amp":[[241,237,236],[229,207,201],[220,177,163],[211,149,127],[202,119,91],[192,88,59],[179,56,38],[157,25,38],[127,14,41],[92,14,33],[60,9,18]],"cmo.tempo":[[255,246,244],[220,223,208],[182,203,175],[141,185,151],[93,167,134],[42,147,127],[17,125,121],[24,101,110],[28,77,97],[26,54,81],[21,29,68]],"cmo.rain":[[238,237,243],[222,211,201],[203,186,152],[161,172,130],[115,157,117],[61,142,110],[12,123,110],[8,99,107],[30,75,95],[37,51,75],[34,27,56]],"cmo.balance":[[24,28,67],[41,62,151],[22,112,188],[89,156,187],[170,194,203],[241,236,235],[220,178,164],[202,118,91],[179,56,38],[126,14,41],[60,9,18]],"cmo.delta":[[17,32,64],[35,67,151],[35,120,163],[78,168,175],[172,206,198],[255,253,205],[216,197,95],[142,162,11],[50,128,31],[16,84,44],[23,35,19]],"cmo.curl":[[21,29,68],[28,77,97],[17,125,121],[93,167,134],[182,203,175],[254,246,245],[230,183,162],[212,119,106],[174,64,96],[118,25,93],[52,13,53]],"cmo.diff":[[8,35,64],[33,77,106],[90,116,135],[141,156,167],[198,202,207],[246,241,240],[206,199,184],[161,150,121],[122,109,65],[75,71,28],[28,34,7]],"cmo.tarn":[[23,35,14],[60,77,13],[132,106,37],[203,132,72],[227,191,163],[252,247,246],[203,204,170],[124,164,144],[46,126,126],[21,79,106],[16,30,79]],"cmo.topo":[[40,26,44],[65,64,122],[64,117,152],[81,167,163],[146,216,164],[13,37,20],[39,81,30],[110,113,60],[179,145,63],[216,195,139],[249,253,228]],"cmo.phase":[[168,120,13],[200,94,55],[220,60,109],[217,38,185],[182,75,237],[125,115,240],[59,140,203],[22,150,150],[25,154,94],[114,143,20],[168,120,13]],"cmr.amber":[[0,0,0],[26,13,15],[61,30,34],[97,43,44],[133,57,45],[166,76,39],[191,102,31],[212,133,30],[228,166,40],[241,204,61],[251,243,87]],"cmr.amethyst":[[0,0,0],[23,15,29],[55,32,74],[86,42,129],[116,43,202],[116,92,225],[121,133,216],[140,167,212],[169,197,216],[207,228,226],[255,255,255]],"cmr.apple":[[0,0,0],[34,9,15],[81,12,29],[126,4,21],[149,53,2],[159,95,6],[159,135,36],[147,176,79],[140,211,144],[197,233,209],[255,255,255]],"cmr.arctic":[[0,0,0],[17,18,27],[38,42,66],[53,64,108],[61,88,158],[49,118,198],[71,147,201],[116,175,204],[160,200,213],[208,227,230],[255,255,255]],"cmr.bubblegum":[[7,34,56],[32,45,95],[64,49,141],[101,43,179],[139,37,199],[174,45,198],[202,65,186],[224,94,170],[237,125,161],[246,158,161],[252,189,166]],"cmr.chroma":[[0,0,0],[2,16,57],[62,1,117],[113,2,110],[162,13,75],[186,68,30],[185,121,0],[157,174,33],[59,226,99],[166,242,201],[255,255,255]],"cmr.cosmic":[[0,0,0],[21,12,28],[51,24,78],[80,22,142],[96,19,219],[75,86,229],[50,127,221],[31,161,216],[15,191,215],[8,224,213],[95,253,208]],"cmr.dusk":[[0,0,0],[8,15,27],[4,42,50],[31,63,53],[65,81,49],[108,94,39],[153,102,35],[201,106,49],[238,115,91],[242,153,152],[241,191,202]],"cmr.eclipse":[[0,0,0],[13,17,28],[22,43,68],[5,70,93],[47,94,98],[86,114,105],[122,135,110],[166,154,109],[212,172,101],[241,203,87],[254,242,85]],"cmr.ember":[[0,0,0],[19,13,27],[59,24,53],[100,26,67],[144,18,68],[186,13,52],[214,54,29],[231,98,7],[240,139,3],[244,182,31],[241,225,62]],"cmr.emerald":[[0,0,0],[12,18,10],[28,43,24],[40,67,33],[48,92,40],[46,119,47],[18,146,71],[24,171,112],[57,195,152],[89,221,195],[119,247,238]],"cmr.fall":[[36,5,29],[74,3,62],[112,11,64],[140,39,53],[165,68,40],[184,99,21],[195,133,0],[193,172,54],[199,203,130],[222,229,199],[255,255,255]],"cmr.flamingo":[[0,0,0],[30,13,6],[73,26,12],[116,30,20],[164,18,41],[205,12,90],[222,72,145],[226,126,185],[228,172,211],[236,216,233],[255,255,255]],"cmr.freeze":[[0,0,0],[20,16,30],[45,37,77],[64,55,133],[69,76,199],[37,114,224],[34,149,219],[82,180,214],[139,205,215],[201,229,230],[255,255,255]],"cmr.gem":[[65,2,6],[92,1,40],[113,12,86],[128,27,135],[137,47,187],[136,76,232],[121,113,252],[97,149,254],[75,179,250],[54,208,246],[68,233,238]],"cmr.ghostlight":[[0,0,0],[17,16,29],[47,32,74],[83,45,97],[106,73,97],[123,101,102],[137,129,108],[148,161,112],[155,194,110],[188,224,88],[254,242,85]],"cmr.gothic":[[0,0,0],[17,17,35],[43,34,93],[81,29,166],[131,12,203],[173,49,197],[202,89,181],[200,141,204],[201,183,234],[215,222,247],[255,255,255]],"cmr.horizon":[[13,52,11],[4,73,48],[14,93,85],[33,111,122],[56,128,164],[90,143,209],[134,155,243],[179,172,247],[208,197,242],[232,226,244],[255,255,255]],"cmr.jungle":[[0,0,0],[8,22,16],[14,51,31],[10,79,38],[5,108,34],[44,135,6],[104,154,48],[145,176,101],[181,199,152],[218,226,206],[255,255,255]],"cmr.lavender":[[0,0,0],[24,6,21],[58,6,66],[77,11,121],[71,57,145],[59,92,146],[52,119,144],[42,146,142],[23,172,133],[44,199,108],[119,219,70]],"cmr.lilac":[[0,0,0],[1,19,27],[7,40,80],[55,49,125],[97,63,131],[130,83,124],[156,107,117],[174,136,109],[179,170,110],[174,208,116],[146,251,122]],"cmr.neon":[[167,12,3],[182,16,77],[196,5,151],[195,41,220],[169,100,251],[131,143,253],[83,175,243],[23,202,218],[30,222,185],[81,241,134],[142,254,21]],"cmr.neutral":[[0,0,0],[19,19,19],[44,43,43],[67,66,66],[90,89,89],[113,113,113],[138,137,137],[164,164,163],[192,192,192],[222,222,222],[255,255,255]],"cmr.nuclear":[[0,0,0],[15,17,9],[32,42,24],[44,65,37],[51,91,49],[54,117,59],[52,143,67],[46,172,69],[56,200,63],[100,227,48],[151,252,26]],"cmr.ocean":[[28,0,65],[37,32,92],[38,61,117],[37,87,133],[41,113,146],[56,139,156],[80,164,165],[119,188,174],[167,208,193],[213,230,222],[255,255,255]],"cmr.pepper":[[62,5,23],[93,0,25],[119,18,14],[134,47,5],[144,74,1],[150,101,7],[151,127,22],[148,155,36],[138,183,46],[114,214,46],[45,246,23]],"cmr.rainforest":[[0,0,0],[33,5,37],[62,3,116],[40,60,146],[1,100,130],[14,132,116],[54,160,88],[138,181,17],[214,191,74],[251,214,176],[255,255,255]],"cmr.sapphire":[[0,0,0],[15,16,22],[35,37,56],[52,57,91],[68,77,133],[78,98,178],[87,125,187],[107,152,194],[129,178,204],[152,207,217],[173,237,232]],"cmr.savanna":[[0,0,0],[8,20,31],[2,50,54],[24,76,58],[62,100,51],[110,120,33],[163,133,9],[214,147,54],[233,178,129],[242,217,199],[255,255,255]],"cmr.sepia":[[0,0,0],[21,15,22],[52,34,51],[84,50,72],[115,68,84],[144,90,91],[167,115,100],[184,145,116],[198,176,142],[211,210,178],[228,244,218]],"cmr.sunburst":[[0,0,0],[29,12,24],[73,21,47],[118,22,53],[165,14,43],[203,39,13],[218,93,22],[223,141,69],[224,183,124],[229,223,193],[255,255,255]],"cmr.swamp":[[0,0,0],[18,17,29],[29,45,66],[26,73,87],[20,103,99],[23,132,103],[54,160,99],[119,183,102],[171,203,149],[214,227,204],[255,255,255]],"cmr.torch":[[0,0,0],[4,20,39],[5,38,117],[71,45,151],[117,65,143],[164,80,135],[211,92,115],[250,118,81],[252,169,101],[244,217,180],[255,255,255]],"cmr.toxic":[[0,0,0],[18,13,30],[36,32,82],[27,62,101],[27,90,104],[25,117,105],[9,144,99],[25,173,77],[91,197,25],[169,212,78],[217,230,145]],"cmr.tree":[[0,0,0],[23,11,5],[56,27,8],[81,45,2],[91,72,18],[93,100,47],[91,126,73],[77,156,93],[49,186,93],[36,217,73],[55,246,17]],"cmr.tropical":[[144,14,165],[175,6,138],[202,20,100],[215,56,61],[216,94,30],[207,130,3],[187,162,24],[150,194,69],[94,221,125],[36,240,192],[68,252,252]],"cmr.voltage":[[0,0,0],[29,11,24],[70,21,65],[107,16,121],[129,33,187],[130,82,229],[119,127,250],[111,169,253],[145,201,242],[202,228,241],[255,255,255]],"cmr.fusion":[[39,4,18],[115,6,51],[173,58,36],[203,125,48],[216,194,146],[255,255,255],[155,209,222],[73,158,222],[83,91,213],[86,12,154],[24,10,50]],"cmr.guppy":[[250,145,79],[254,80,25],[231,2,52],[179,28,87],[129,30,104],[88,7,114],[100,30,191],[79,80,242],[37,132,218],[40,168,200],[30,204,191]],"cmr.holly":[[20,20,1],[47,69,13],[44,122,35],[112,162,108],[186,203,177],[255,255,255],[220,189,187],[194,126,126],[162,66,87],[96,38,70],[31,9,34]],"cmr.iceburn":[[148,241,243],[75,186,228],[30,129,220],[60,78,142],[34,37,59],[0,0,0],[58,28,32],[127,53,44],[185,92,28],[222,152,25],[245,222,69]],"cmr.pride":[[22,14,40],[42,53,115],[36,104,126],[87,146,126],[154,187,107],[249,221,41],[242,148,13],[214,84,43],[167,28,72],[101,7,81],[31,10,26]],"cmr.prinsenvlag":[[170,82,69],[205,104,37],[220,144,16],[218,187,102],[228,223,185],[255,255,255],[190,232,229],[121,208,227],[90,175,235],[103,136,229],[108,108,162]],"cmr.redshift":[[148,241,243],[75,186,228],[30,129,220],[60,78,142],[34,37,59],[0,0,0],[58,29,24],[130,50,47],[199,69,98],[228,135,180],[252,205,252]],"cmr.seaweed":[[145,241,251],[89,184,224],[73,128,189],[65,80,120],[35,38,53],[0,0,0],[32,42,24],[51,91,49],[52,143,67],[56,200,63],[151,252,26]],"cmr.viola":[[28,7,48],[81,32,137],[84,100,174],[87,159,202],[159,208,216],[255,255,255],[217,187,217],[198,114,189],[175,35,125],[115,11,43],[40,4,12]],"cmr.waterlily":[[32,2,55],[75,39,134],[79,102,169],[83,160,198],[162,208,214],[255,255,255],[172,208,176],[87,169,89],[72,117,19],[60,65,20],[27,16,6]],"cmr.watermelon":[[151,252,26],[56,200,63],[52,143,67],[51,91,49],[32,42,24],[0,0,0],[54,31,30],[119,59,59],[186,84,103],[221,141,176],[251,207,242]],"cmr.wildfire":[[147,253,42],[57,197,117],[30,141,130],[13,88,124],[33,35,74],[0,0,0],[62,20,51],[143,19,76],[199,75,52],[237,141,24],[252,220,31]],"cmr.copper":[[0,0,0],[34,42,55],[51,91,115],[95,140,143],[159,188,171],[238,241,206],[208,172,135],[169,113,103],[114,68,87],[51,35,50],[0,0,0]],"cmr.copper_s":[[238,241,206],[208,172,135],[169,113,103],[114,68,87],[51,35,50],[0,0,0],[34,42,55],[51,91,115],[95,140,143],[159,188,171],[236,240,205]],"cmr.emergency":[[0,0,0],[5,49,67],[53,88,170],[152,115,233],[208,181,226],[255,255,255],[225,175,203],[209,92,125],[147,54,27],[64,35,3],[0,0,0]],"cmr.emergency_s":[[255,255,255],[225,175,203],[209,92,125],[147,54,27],[64,35,3],[0,0,0],[5,49,67],[53,88,170],[152,115,233],[208,181,226],[254,253,254]],"cmr.infinity":[[88,7,113],[83,55,172],[40,106,176],[13,142,158],[71,171,136],[152,192,106],[181,144,29],[190,93,11],[171,52,55],[131,32,91],[89,8,113]],"cmr.infinity_s":[[152,192,106],[181,144,29],[190,93,11],[171,52,55],[131,32,91],[88,7,113],[83,55,172],[40,106,176],[13,142,158],[71,171,136],[150,191,106]],"cmr.seasons":[[0,0,0],[39,38,83],[30,99,121],[55,158,125],[164,205,141],[255,255,255],[202,193,123],[193,114,48],[149,47,62],[70,22,57],[0,0,0]],"cmr.seasons_s":[[255,255,255],[202,193,123],[193,114,48],[149,47,62],[70,22,57],[0,0,0],[39,38,83],[30,99,121],[55,158,125],[164,205,141],[253,254,253]]};
const COLORMAP_GROUPS = [{"label":"Perceptually uniform sequential","source":"core","names":["viridis","plasma","inferno","magma","cividis"]},{"label":"Sequential","source":"core","names":["Greys","Purples","Blues","Greens","Oranges","Reds","YlOrBr","YlOrRd","OrRd","PuRd","RdPu","BuPu","GnBu","PuBu","YlGnBu","PuBuGn","BuGn","YlGn"]},{"label":"Sequential (2)","source":"core","names":["binary","gist_yarg","gist_gray","gray","bone","pink","spring","summer","autumn","winter","cool","Wistia","hot","afmhot","gist_heat","copper"]},{"label":"Diverging","source":"core","names":["PiYG","PRGn","BrBG","PuOr","RdGy","RdBu","RdBu_r","RdYlBu","RdYlGn","Spectral","coolwarm","bwr","seismic","berlin","managua","vanimo"]},{"label":"Cyclic","source":"core","names":["twilight","twilight_shifted","hsv"]},{"label":"Miscellaneous","source":"core","names":["flag","prism","ocean","gist_earth","terrain","gist_stern","gnuplot","gnuplot2","CMRmap","cubehelix","brg","gist_rainbow","rainbow","jet","turbo","nipy_spectral","gist_ncar"]},{"label":"cmocean: Sequential","source":"cmocean","names":["cmo.thermal","cmo.haline","cmo.solar","cmo.ice","cmo.gray","cmo.oxy","cmo.deep","cmo.dense","cmo.algae","cmo.matter","cmo.turbid","cmo.speed","cmo.amp","cmo.tempo","cmo.rain"]},{"label":"cmocean: Diverging","source":"cmocean","names":["cmo.balance","cmo.delta","cmo.curl","cmo.diff","cmo.tarn","cmo.topo"]},{"label":"cmocean: Cyclic","source":"cmocean","names":["cmo.phase"]},{"label":"cmasher: Sequential","source":"cmasher","names":["cmr.amber","cmr.amethyst","cmr.apple","cmr.arctic","cmr.bubblegum","cmr.chroma","cmr.cosmic","cmr.dusk","cmr.eclipse","cmr.ember","cmr.emerald","cmr.fall","cmr.flamingo","cmr.freeze","cmr.gem","cmr.ghostlight","cmr.gothic","cmr.horizon","cmr.jungle","cmr.lavender","cmr.lilac","cmr.neon","cmr.neutral","cmr.nuclear","cmr.ocean","cmr.pepper","cmr.rainforest","cmr.sapphire","cmr.savanna","cmr.sepia","cmr.sunburst","cmr.swamp","cmr.torch","cmr.toxic","cmr.tree","cmr.tropical","cmr.voltage"]},{"label":"cmasher: Diverging","source":"cmasher","names":["cmr.fusion","cmr.guppy","cmr.holly","cmr.iceburn","cmr.pride","cmr.prinsenvlag","cmr.redshift","cmr.seaweed","cmr.viola","cmr.waterlily","cmr.watermelon","cmr.wildfire"]},{"label":"cmasher: Cyclic","source":"cmasher","names":["cmr.copper","cmr.copper_s","cmr.emergency","cmr.emergency_s","cmr.infinity","cmr.infinity_s","cmr.seasons","cmr.seasons_s"]}];

// ---- core/bus.js
// app.bus: minimal event bus. on() returns an unsubscribe function; a throwing handler never stops the others.
function createBus() {
  const handlers = new globalThis.Map();
  const bus = {
    on(event, fn) {
      if (!handlers.has(event)) handlers.set(event, new Set());
      handlers.get(event).add(fn);
      return () => bus.off(event, fn);
    },
    off(event, fn) { handlers.get(event)?.delete(fn); },
    emit(event, payload) {
      for (const fn of [...(handlers.get(event) || [])]) {
        try { fn(payload); } catch (e) { console.error(`handler for "${event}" failed`, e); }
      }
    },
  };
  return bus;
}

// ---- core/config.js
// Store handling, ported from the reference: noHeadSuffixReads, storeError, publicStoreUrl, attrLimits, browserConfig,
// openStore (D-35: the browser reads the store, no Python config path). Nothing here touches the DOM.

// Serve suffix ranges (shard indexes) as one `Range: bytes=-N` GET instead of HEAD + explicit range (D-56). Idempotent.
function noHeadSuffixReads(FetchStore) {
  const proto = FetchStore.prototype;
  if (proto.noHead) return;
  const getRange = proto.getRange;
  proto.getRange = function (key, range, opts) {
    if (!('suffixLength' in range)) return getRange.call(this, key, range, opts);
    return this.get(key, {...opts, headers: {...opts?.headers, Range: `bytes=-${range.suffixLength}`}});
  };
  proto.noHead = true;
}

function storeError(message, hint = '', code = 'error') {
  const e = new Error(message); e.hint = hint; e.code = code; return e;
}

// Config of a viewer without a store: global EPSG:4326 view, basemaps only.
function blankConfig() {
  return {title: '', source: '', store_url: '', endpoint: '', variables: [], dims: {}, attrs: {}, levels: 0,
    bbox: [-180, -90, 180, 90], global: true,
    crs: {code: 'EPSG:4326', name: 'EPSG:4326', proj4: null, projection: null}};
}

// Static hosting reads public stores in the browser. Returns the https URL of a store given as s3://, gs://, https://
// or a path relative to the page.
function publicStoreUrl(src, endpoint) {
  if (/^file:/i.test(src)) throw storeError('Serve the store over HTTP to open it here.',
    'Use a URL or a path relative to this page; local filesystem paths cannot be read by a web page.', 'unsupported_scheme');
  if (/^s3:\/\//i.test(src)) {
    const u = new URL(src);
    const base = endpoint ? endpoint.replace(/\/$/, '') + '/' + u.hostname : `https://${u.hostname}.s3.amazonaws.com`;
    return base + u.pathname;
  }
  if (/^(gs|gcs):\/\//i.test(src)) { const u = new URL(src); return `https://storage.googleapis.com/${u.hostname}${u.pathname}`; }
  const u = new URL(src, location.href);
  if (!/^https?:$/.test(u.protocol)) throw storeError('Use an HTTP(S) GeoZarr URL.', '', 'unsupported_scheme');
  return u.href.replace(/\/$/, '');
}

// The spec's `relay` is a URL prefix (e.g. "/proxy/", scripts/serve.py) for stores without CORS rules: it is put in
// front of cross-origin http(s) store URLs; same-origin URLs are left alone.
function relayUrl(url, relay) {
  if (!relay || !/^https?:\/\//i.test(url)) return url;
  if (new URL(url).origin === location.origin) return url;
  return relay + url;
}

// D-43 (mirrors preview._attr_limits): the vmin/vmax attrs override the computed limits; if both are given
// and vmax <= vmin, both are ignored, otherwise vmax <= vmin becomes vmin + 1.
const attrNum = (x) => typeof x === 'number' && Number.isFinite(x) ? x : null;
function attrLimits(a, vmin, vmax, name) {
  let lo = attrNum(a.vmin), hi = attrNum(a.vmax);
  if (lo !== null && hi !== null && hi <= lo) {
    console.warn(`${name}: vmax <= vmin in attrs (${lo} >= ${hi}), ignoring both`);
    lo = hi = null;
  }
  if (lo !== null) vmin = lo;
  if (hi !== null) vmax = hi;
  return [vmin, vmax <= vmin ? vmin + 1 : vmax];
}
// D-53 (mirrors preview._has_attr_limits): true when attrLimits takes a limit from the attrs.
function hasAttrLimits(a) {
  const lo = attrNum(a.vmin), hi = attrNum(a.vmax);
  if (lo !== null && hi !== null && hi <= lo) return false;
  return lo !== null || hi !== null;
}

// D-55: attribute display values, mirroring `_attr_value` in preview.py.
function attrValue(x) {
  let out;
  if (typeof x === 'boolean') out = x ? 'True' : 'False';
  else if (typeof x === 'number') out = Number.isFinite(x) ? x : String(x);
  else if (typeof x === 'string') out = x;
  else if (Array.isArray(x)) out = x.map((i) => typeof i === 'string' ? i : (i && typeof i === 'object' ? JSON.stringify(i) : String(i))).join(', ');
  else if (x && typeof x === 'object') out = JSON.stringify(x);
  else out = x == null ? 'None' : String(x);
  return typeof out === 'string' && out.length > 2000 ? out.slice(0, 2000) + '…' : out;
}

// Reads the GeoZarr pyramid metadata in the browser and returns the dataset config (variables with default limits,
// dims with ISO labels, crs, bbox). `relay`: see relayUrl.
async function browserConfig(src, endpoint, relay) {
  const url = new URL(relayUrl(publicStoreUrl(src, endpoint), relay), location.href).href;
  const metadata = async (path) => {
    const res = await fetch(`${url}/${path ? path + '/' : ''}zarr.json`);
    if (!res.ok) throw storeError(`Cannot read GeoZarr metadata (${res.status}).`, 'Check the store URL and public access.', 'not_found');
    return res.json();
  };
  try {
    const root = await metadata('');
    const attrs = root.attributes || {};
    const layout = attrs.multiscales?.layout;
    if (!layout?.length) throw storeError('This store has no GeoZarr multiscales layout.', 'Open a GeoZarr pyramid, rather than a plain Zarr array.', 'not_pyramid');
    const level = String(layout.at(-1).asset);
    const group = root.consolidated_metadata?.metadata?.[level] || await metadata(level);
    const sdims = group.attributes?.['spatial:dimensions'] || attrs['spatial:dimensions'];
    if (!sdims?.length) throw storeError('Missing spatial dimensions.', '', 'invalid_pyramid');
    // Consolidated member metadata avoids directory listings, which static servers cannot provide.
    const members = group.consolidated_metadata?.metadata || Object.fromEntries(
      Object.entries(root.consolidated_metadata?.metadata || {})
        .filter(([path]) => path.startsWith(level + '/'))
        .map(([path, value]) => [path.slice(level.length + 1), value]));
    const order = attrs['geozarr_pyramid_maker:variables'] || [];
    const names = [...new Set([...order, ...Object.keys(members).filter((n) => !n.includes('/'))])];
    if (!names.length) throw storeError('No array metadata is available.', 'Use a store with consolidated metadata or named variables.', 'invalid_pyramid');
    const zarr = await import(ZARRITA_URL);
    noHeadSuffixReads(zarr.FetchStore);
    const zr = zarr.root(new zarr.FetchStore(url, {useSuffixRequest: true}));
    const variables = [], dims = {};
    for (const name of names) {
      const meta = members[name] || await metadata(`${level}/${name}`);
      const dn = meta.dimension_names || [];
      if (meta.node_type !== 'array' || !sdims.every((d) => dn.includes(d))) continue;
      const a = meta.attributes || {}, extra = dn.filter((d) => !sdims.includes(d));
      const arr = await zarr.open(zr.resolve(`${level}/${name}`), {kind: 'array'});
      const sample = await zarr.get(arr, dn.map((d) => extra.includes(d) ? 0 : null));
      const values = Array.from(sample.data, Number).filter((v) => Number.isFinite(v) && v !== meta.fill_value).sort((a, b) => a - b);
      const categorical = a.resampling_method === 'mode' || /^(u?int|bool)/.test(meta.data_type);
      const quantile = (p) => {
        const i = (values.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i);
        return values[lo] + (values[hi] - values[lo]) * (i - lo);
      };
      let vmin = values.length ? (categorical ? values[0] : quantile(.02)) : 0;
      let vmax = values.length ? (categorical ? values.at(-1) : quantile(.98)) : 1;
      if (vmax <= vmin) vmax = vmin + 1;
      // D-43: plot defaults from the attrs (categorical ignores them); `cmap` is resolved later, in defaultStyle.
      if (!categorical) [vmin, vmax] = attrLimits(a, vmin, vmax, name);
      const v = {name, dims: extra, categorical, units: String(a.units || ''), long_name: String(a.long_name || ''),
        display_name: String(a.long_name || name), vmin, vmax,
        limits_from_attrs: !categorical && hasAttrLimits(a),
        attrs: Object.fromEntries(Object.entries(a).map(([k, x]) => [k, attrValue(x)]))};
      if (!categorical && a.cmap != null) v.cmap = String(a.cmap);
      if (categorical) {
        v.values = [...new Set(values)].slice(0, 20);
        const palette = [[31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189],[140,86,75],[227,119,194],[127,127,127],[188,189,34],[23,190,207]];
        v.palette = v.values.map((n, i) => [n, palette[i % palette.length]]);
      }
      variables.push(v);
      for (const d of extra) {
        if (dims[d]) continue;
        const size = meta.shape[dn.indexOf(d)];
        dims[d] = {size, labels: null, iso: null};
        if (size > 5000) continue;
        try {
          const coord = await zarr.open(zr.resolve(`${level}/${d}`), {kind: 'array'});
          const data = Array.from((await zarr.get(coord)).data);
          const units = coord.attrs.units || '';
          const match = /^(seconds?|minutes?|hours?|days?) since (.+)$/i.exec(units);
          const calendar = coord.attrs.calendar || 'standard';
          if (match && ['standard', 'gregorian', 'proleptic_gregorian'].includes(calendar)) {
            const scale = {second: 1000, minute: 60000, hour: 3600000, day: 86400000}[match[1].toLowerCase().replace(/s$/, '')];
            let epoch = match[2].trim().replace(' ', 'T');
            if (!epoch.includes('T')) epoch += 'T00:00:00';
            if (!/(Z|[+-]\d{2}:?\d{2})$/i.test(epoch)) epoch += 'Z';
            const origin = Date.parse(epoch);
            if (Number.isFinite(origin)) dims[d].iso = data.map((x) => new Date(origin + Number(x) * scale).toISOString());
          }
          dims[d].labels = dims[d].iso || data.map(String);
        } catch (_) { /* index controls work without coordinate arrays */ }
      }
    }
    if (!variables.length) throw storeError('No plottable spatial variables found.', '', 'invalid_pyramid');
    const code = attrs['proj:code'];
    const definition = attrs['proj:wkt2'] || attrs['proj:projjson'];
    if (!code && !definition) throw storeError('Missing CRS metadata.', '', 'invalid_pyramid');
    const name = code || 'GPM:custom';
    const builtin = ['EPSG:4326', 'EPSG:3857'].includes(code);
    if (!builtin && !definition) throw storeError(`No CRS definition for ${code}.`, 'Include proj:wkt2 metadata.', 'invalid_pyramid');
    const bbox = attrs['spatial:bbox'] || group.attributes?.['spatial:bbox'];
    if (!Array.isArray(bbox) || bbox.length !== 4) throw storeError('Missing spatial bounding box.', '', 'invalid_pyramid');
    return {...blankConfig(), title: src.split('/').filter(Boolean).at(-1), source: src, store_url: url, endpoint: endpoint || '',
      variables, dims, attrs: Object.fromEntries(Object.entries(attrs).filter(([k]) =>
        !['multiscales', 'zarr_conventions', 'geozarr_pyramid_maker:variables'].includes(k) && !/^(proj|spatial):/.test(k))
        .map(([k, x]) => [k, attrValue(x)])),
      levels: layout.length, bbox, global: code === 'EPSG:4326' && Math.abs(bbox[2] - bbox[0] - 360) < 1e-6,
      crs: {code, name, proj4: builtin ? null : definition, projection: builtin ? null : name}};
  } catch (e) {
    if (e.code) throw e;
    throw storeError(`Cannot open the GeoZarr store: ${e.message}`,
      'Check its metadata and codecs. Remote stores must allow CORS and HTTP Range requests (or use the spec option `relay`).', 'browser_blocked');
  }
}

// Opens a store: validates it by reading its metadata.
const openStore = (src, endpoint, relay) => browserConfig(src, endpoint, relay);

// ---- core/style.js
// Colourmaps, per-variable style state, colour expressions, ticks. Ported from the reference: cmaps/cmapGroups,
// resolveCmap, defaultStyle, loadStyle, styleOf, stopsOf, effectiveLog, gradientCss, niceTicks, niceLogTicks, colorExpr.
// Data (COLORMAPS, COLORMAP_GROUPS) lives in colormaps.js. The pure functions come first; createStyle builds app.style.

const rgba = (c) => [c[0], c[1], c[2], 1];
const cmapLabel = (name) => name.replace(/^cm[or]\./, '');

// D-43: the variable's cmap (from the attrs): an exact key, else "<key>_r" = the key reversed, else viridis.
function resolveCmap(v, cmaps) {
  const name = v.cmap;
  if (name == null) return {cmap: 'viridis', reversed: !!v.reversed};
  if (cmaps[name]) return {cmap: name, reversed: !!v.reversed};
  if (/_r$/.test(name) && cmaps[name.slice(0, -2)]) return {cmap: name.slice(0, -2), reversed: !v.reversed};
  console.warn(`${v.name}: unknown cmap ${JSON.stringify(name)}, using viridis`);
  return {cmap: 'viridis', reversed: !!v.reversed};
}
const defaultStyle = (v, cmaps) => ({...resolveCmap(v, cmaps), vmin: v.vmin, vmax: v.vmax});
const stopsOf = (st, cmaps) => { const s = cmaps[st.cmap]; return st.reversed ? [...s].reverse() : s; };
// D-54: the log scale needs a positive Min; the stored flag is kept while it is not.
const effectiveLog = (st) => !!st.log && st.vmin > 0;
const gradientCss = (stops) => 'linear-gradient(to right,' +
  stops.map((c, i) => `rgb(${c[0]},${c[1]},${c[2]}) ${100 * i / (stops.length - 1)}%`).join(',') + ')';

// Groups to list in a picker: a group whose source is switched off still lists the current map (D-42), so a select
// never loses its value. show: {core, cmocean, cmasher} -> boolean. Returns [{label, names}].
function listedGroups(groups, show, current) {
  const out = [];
  for (const g of groups) {
    const names = show[g.source] ? g.names : g.names.filter((n) => n === current);
    if (names.length) out.push({label: g.label, source: g.source, names});
  }
  return out;
}

// Mirrors preview._nice_ticks (Python); keep the two in sync. Shared cases:
//   (0, 1) -> 0,.2,.4,.6,.8,1    (280, 470) -> 300,350,400,450    (-3.3, 31.7) -> 0,10,20,30
//   (0.0013, 0.0021) -> .0014,.0016,.0018,.002    (1e6, 9.3e6) -> 2e6,4e6,6e6,8e6
//   (5, 5.5) -> 5,5.1,...,5.5    (2, 3) -> 2,2.2,...,3
function niceTicks(vmin, vmax) {
  const clean = (x) => +Number(x).toPrecision(10);
  const span = vmax - vmin;
  if (!(span > 0 && Number.isFinite(span))) return [clean(vmin)];
  const base = Math.floor(Math.log10(span));
  let best = null;
  for (let exp = base - 2; exp < base + 2; exp++) {
    for (const mant of [1, 2, 5]) {
      const step = mant * 10 ** exp;
      const k0 = Math.ceil(vmin / step - 1e-9), k1 = Math.floor(vmax / step + 1e-9);
      const n = k1 - k0 + 1;
      if (n < 4 || n > 6) continue;
      const score = Math.abs(n - 5) - step * 1e-12 / span;
      if (!best || score < best.score) {
        best = {score, ticks: Array.from({length: n}, (_, i) => clean((k0 + i) * step))};
      }
    }
  }
  return best ? best.ticks : Array.from({length: 5}, (_, i) => clean(vmin + span * i / 4));
}

// Ticks of a log scale: 1-2-5 x 10^k inside [vmin, vmax], only 1 x 10^k when that gives more than 6, vmin and vmax
// when fewer than 2 fit. Shared cases: (1, 10) -> 1,2,5,10    (3, 80) -> 5,10,20,50    (1, 1000) -> 1,10,100,1000
function niceLogTicks(vmin, vmax) {
  const clean = (x) => +Number(x).toPrecision(10);
  const within = (mants) => {
    const out = [];
    for (let k = Math.floor(Math.log10(vmin)); k <= Math.ceil(Math.log10(vmax)); k++) {
      for (const m of mants) {
        const t = m * 10 ** k;
        if (t >= vmin * (1 - 1e-9) && t <= vmax * (1 + 1e-9)) out.push(clean(t));
      }
    }
    return out;
  };
  let ticks = within([1, 2, 5]);
  if (ticks.length > 6) ticks = within([1]);
  return ticks.length >= 2 ? ticks : [clean(vmin), clean(vmax)];
}

// OpenLayers colour expression of a variable: categorical `match`, else `interpolate` over the colourmap stops.
function colorExpr(v, st, cmaps) {
  if (v.categorical) {
    const expr = ['match', ['band', 1]];
    for (const [val, c] of v.palette) expr.push(val, rgba(c));
    expr.push([0, 0, 0, 0]);
    return expr;
  }
  const stops = stopsOf(st, cmaps);
  const expr = ['interpolate', ['linear'], ['band', 1]];
  const n = stops.length;
  // no log operator in OL expressions: the log scale puts the stops at geometric values instead (D-54)
  const at = effectiveLog(st)
    ? (f) => st.vmin * (st.vmax / st.vmin) ** f
    : (f) => st.vmin + (st.vmax - st.vmin) * f;
  stops.forEach((c, i) => expr.push(at(i / (n - 1)), rgba(c)));
  return expr;
}

const SOURCES = ['core', 'cmocean', 'cmasher'];
const SOURCE_KEYS = {core: 'cmaps-matplotlib', cmocean: 'cmaps-cmocean', cmasher: 'cmaps-cmasher'};

// app.style: per-variable style state (cmap, reversed, vmin, vmax, log). `explicit` marks ranges set by the user, the
// URL or the attrs (D-53): the automatic Auto on add skips those. Changes emit `style:changed(name)`.
function createStyle(app) {
  const cmaps = COLORMAPS, groups = COLORMAP_GROUPS;
  const styles = {}, explicit = new Set();
  const storeKey = (name) => `style:${app.cfg.title}:${name}`;

  // D-42/D-47: each source (core = matplotlib, cmocean, cmasher) is listed only while on; matplotlib defaults to on,
  // the others to off; at least one source stays on. Looking a map up (`cmaps`) never depends on it.
  const show = Object.fromEntries(SOURCES.map((src) => {
    const stored = app.storage.get(SOURCE_KEYS[src]);
    return [src, src === 'core' ? stored !== '0' : stored === '1'];
  }));
  if (!Object.values(show).some(Boolean)) show.core = true;  // stored state with no source on (not written back)

  const varOf = (name) => app.varOf(name);
  function load(v) {
    try {
      const raw = app.storage.get(storeKey(v.name));
      const s = raw ? JSON.parse(raw) : null;
      if (s && cmaps[s.cmap] && Number.isFinite(s.vmin) && Number.isFinite(s.vmax) && s.vmin < s.vmax) {
        explicit.add(v.name);
        return {cmap: s.cmap, reversed: !!s.reversed, vmin: s.vmin, vmax: s.vmax, log: !!s.log};
      }
    } catch (_) { /* corrupt entry: use the defaults */ }
    return defaultStyle(v, cmaps);
  }

  const style = {
    cmaps, groups, sources: show, cmapLabel,
    // current style of a variable (a copy; change it with set)
    get(name) {
      const v = varOf(name);
      if (!v) return null;
      return {...(styles[name] || (styles[name] = load(v)))};
    },
    // Merge a patch. `explicit` (default true) marks the range as user-set and, with `persist`, stores it. Rejects (false)
    // an empty/invalid range or an unknown colourmap. The automatic Auto on add passes {explicit: false}.
    set(name, patch, {explicit: mark = true, persist = true} = {}) {
      const v = varOf(name);
      if (!v || v.categorical) return false;
      const next = {...style.get(name), ...patch};
      if (!cmaps[next.cmap] || !(Number.isFinite(next.vmin) && Number.isFinite(next.vmax) && next.vmin < next.vmax)) return false;
      styles[name] = next;
      if (mark) {
        explicit.add(name);
        if (persist) app.storage.set(storeKey(name), JSON.stringify(next));
      }
      app.bus.emit('style:changed', name);
      app.url.sync();
      return true;
    },
    isExplicit: (name) => explicit.has(name),
    stopsOf: (st) => stopsOf(st, cmaps),
    gradientCss, effectiveLog, niceTicks, niceLogTicks,
    colorExpr(name) {
      const v = varOf(name);
      return colorExpr(v, v.categorical ? null : style.get(name), cmaps);
    },
    // groups to list in a picker for the current map (see listedGroups)
    listedGroups: (current) => listedGroups(groups, show, current),
    // Switch a colourmap source on/off (D-42); the last one cannot be switched off. Emits `cmaps:changed`.
    setSourceShown(source, on) {
      if (!SOURCES.includes(source)) return false;
      if (!on && !SOURCES.some((s) => s !== source && show[s])) return false;
      show[source] = !!on;
      app.storage.set(SOURCE_KEYS[source], on ? '1' : '0');
      app.bus.emit('cmaps:changed', {...show});
      return true;
    },
  };
  return style;
}

// ---- core/layers.js
// Map and data layers. Ported from the reference: normLon, shiftGrid, makeLayers (world copies, D-26), viewFor, makeLayer,
// addVariable/removeVariable/restackLayers, toggleVar/toggleVisibility/moveVariable, setDim, autoOnAdd (D-53),
// initCenterLon, initBasemap (layers only; the selector is the basemap component). The sidebar/card DOM is not here.

const normLon = (x) => { const l = ((x + 180) % 360 + 360) % 360 - 180; return Number.isFinite(l) ? l : 0; };

// Global data (any centre longitude, 0 included): two extra copies of the data layer, shifted by
// ±360° (their tile grids are offset, so chunk indices are unchanged). Not wrapX (see D-24, D-26).
function shiftGrid(grid, dx) {
  const e = grid.getExtent();
  return new WMTSTileGrid({
    extent: [e[0] + dx, e[1], e[2] + dx, e[3]],
    origins: grid.getResolutions().map((_, z) => { const o = grid.getOrigin(z); return [o[0] + dx, o[1]]; }),
    resolutions: grid.getResolutions(),
    matrixIds: grid.getMatrixIds(),
    tileSizes: grid.getResolutions().map((_, z) => grid.getTileSize(z)),
  });
}

// View options for the map: global data is bounded to one world centred on `lon`; x, y, z in the URL pin the view.
function viewOptions(opts, cfg, lon, params) {
  const x = Number(params.get('x')), y = Number(params.get('y')), z = Number(params.get('z'));
  const pinned = params.has('z') && Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z);
  const b = cfg.bbox;
  if (cfg.global) {
    opts = {...opts, center: [lon, (b[1] + b[3]) / 2], extent: [lon - 180, b[1], lon + 180, b[3]]};
  }
  return pinned ? {...opts, center: [x, y], zoom: z} : opts;
}

// Natural Earth rings that cross the antimeridian (world-atlas keeps them whole, for spherical renderers) streak across
// a planar map. Unwrap: shift each vertex by 360 so no step exceeds 180 degrees. A ring that ends a full turn from its
// start goes round a pole (Antarctica): close it through that pole. Coordinates are [lon, lat] in degrees.
function unwrapRing(ring) {
  const out = [];
  let shift = 0;
  for (let i = 0; i < ring.length; i++) {
    const [x, y] = ring[i];
    if (i) {
      const d = x - ring[i - 1][0];
      if (d > 180) shift -= 360; else if (d < -180) shift += 360;
    }
    out.push([x + shift, y]);
  }
  const first = out[0], last = out[out.length - 1];
  if (Math.abs(last[0] - first[0]) > 180) {
    const pole = out.reduce((s, p) => s + p[1], 0) < 0 ? -90 : 90;
    out.push([last[0], pole], [first[0], pole], [...first]);
  }
  return out;
}

// [id, label]; the label names the tile source
const BASEMAPS = [['light', 'Light (OSM)'], ['dark', 'Dark (OSM)'], ['osm', 'Streets (OSM)'], ['satellite', 'Satellite (Esri)'],
  ['land', 'Land (Natural Earth)'], ['none', 'None']];
const ESRI_IMAGERY = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
// Land (vector): Natural Earth 1:50m land polygons (world-atlas TopoJSON, ~0.5 MB, fetched the first time the layer is
// shown). A plain vector layer, not vector tiles: the view is often EPSG:4326 and OL cannot reproject vector tiles.
// Two layers: the basemap itself is only the --gv-water background (bottom); the land polygons (--gv-land; by default both are --gv-map-bg) are drawn
// above every data layer (zIndex), so land masks the data.
const NE_LAND = 'https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-50m.json';
const NE_ATTRIBUTION = 'Made with <a href="https://www.naturalearthdata.com/" target="_blank">Natural Earth</a>';
const ESRI_ATTRIBUTION = 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community';

function createLayers(app) {
  const {state, bus} = app;
  const vlayers = new globalThis.Map();  // name -> {layers, sources}; only selected, visible variables (hidden ones load nothing, D-52)
  const autoPending = new Set();
  const varOf = (name) => app.varOf(name);
  const selVars = () => state.selected.map(varOf);
  const opacityOf = (name) => (state.opacity[name] === undefined ? 1 : state.opacity[name]);
  const unionDims = () => {  // extra dims of the selected variables: selection order, then each variable's dim order
    const out = [];
    for (const v of selVars()) for (const d of v.dims) if (!out.includes(d)) out.push(d);
    return out;
  };

  // Keyless sources: OSM (Light/Dark are CSS filters on each layer's own canvas, theme.css) and Esri World Imagery.
  const basemaps = {};
  for (const id of ['light', 'dark', 'osm']) basemaps[id] = new TileLayer({className: `bm-${id}`, source: new OSM(), visible: false});
  basemaps.satellite = new TileLayer({className: 'bm-satellite', visible: false,
    source: new XYZ({url: ESRI_IMAGERY, attributions: ESRI_ATTRIBUTION, maxZoom: 19, crossOrigin: 'anonymous'})});
  // flat style object: a Style instance from ol/style.js is another module copy than the layer's (+esm bundles)
  const landSource = new VectorSource({url: NE_LAND, format: new TopoJSON({layers: ['land']}),
    attributions: NE_ATTRIBUTION, wrapX: false});  // own ±360 copies, as the data layers
  // after loading (degree units only): unwrap the rings, then add copies at -360 / +360 so any centre longitude is covered
  landSource.on?.('featuresloadend', () => {
    if (landSource.wrapped || state.map?.getView().getProjection().getUnits() !== 'degrees') return;
    landSource.wrapped = true;
    const copies = [];
    for (const f of landSource.getFeatures()) {
      const g = f.getGeometry();
      const polys = g.getType() === 'Polygon' ? [g.getCoordinates()] : g.getCoordinates();
      const fixed = polys.map((rings) => rings.map(unwrapRing));
      g.setCoordinates(g.getType() === 'Polygon' ? fixed[0] : fixed);
      for (const dx of [-360, 360]) { const c = f.clone(); c.getGeometry().translate(dx, 0); copies.push(c); }
    }
    landSource.addFeatures(copies);
  });
  landSource.on?.('featuresloaderror', () => bus.emit('error', new Error('Land basemap: Natural Earth data unavailable')));
  basemaps.land = new VectorLayer({className: 'bm-land', visible: false, source: new VectorSource()});  // water background
  const landTop = new VectorLayer({className: 'bm-land-top', visible: false, zIndex: 100, style: {'fill-color': '#000'},
    source: landSource});
  // colours from the theme tokens (theme.css defines them, both default to --gv-map-bg), read each time it is shown so they follow the theme
  function showLand() {
    const css = getComputedStyle(document.documentElement);
    basemaps.land.setBackground(css.getPropertyValue('--gv-water').trim());
    landTop.setStyle({'fill-color': css.getPropertyValue('--gv-land').trim()});
  }

  function makeLayer(v) {
    const cfg = app.cfg;
    const dimensions = {};
    for (const n of v.dims) dimensions[n] = clamp(state.dims[n] || 0, 0, cfg.dims[n].size - 1);
    const opts = {url: new URL(cfg.store_url, location.href).href, bands: [v.name], transition: 0};
    if (cfg.crs.projection) opts.projection = cfg.crs.projection;
    if (v.dims.length) opts.dimensions = dimensions;
    const source = new GeoZarr(opts);
    source.addEventListener?.('error', (e) => bus.emit('error', e));
    return {source, layer: new WebGLTileLayer({source, style: {color: app.style.colorExpr(v.name)}, opacity: opacityOf(v.name)})};
  }

  function makeLayers(v) {
    const offsets = app.cfg.global ? [0, -360, 360] : [0];
    const made = offsets.map((dx) => {
      const m = makeLayer(v);
      if (dx) {
        const shift = () => {
          if (m.source.getState() !== 'ready' || m.source.gpmShifted) return;
          m.source.tileGrid = shiftGrid(m.source.tileGrid, dx);
          m.source.gpmShifted = true;
        };
        m.source.on('change', shift);
        shift();
      }
      return m;
    });
    return {sources: made.map((m) => m.source), layers: made.map((m) => m.layer)};
  }

  // Auto range on add (D-53): once the map has rendered the new layer, run Auto, unless the range was set
  // explicitly (stored, URL, attrs, edited) or the variable is categorical. An empty render (no tiles in view yet) is
  // retried on the next two; a variable removed, hidden or edited meanwhile is skipped.
  function autoOnAdd(name) {
    const v = varOf(name);
    if (v.categorical || v.limits_from_attrs || app.style.isExplicit(name) || autoPending.has(name)) return;
    autoPending.add(name);
    const attempt = (left) => state.map.once('rendercomplete', () => {
      const live = state.selected.includes(name) && !state.hidden.has(name) && !app.style.isExplicit(name);
      if (live && !app.frame.autoStyle(name, {auto: true}) && left > 1) return attempt(left - 1);
      autoPending.delete(name);
    });
    attempt(3);
  }

  // Add or remove one variable's layer group; other variables' layers are left untouched.
  // A hidden variable gets no layers; the caller restacks after adding.
  function addVariable(name) {
    if (state.hidden.has(name)) return;
    vlayers.set(name, makeLayers(varOf(name)));
    autoOnAdd(name);  // runs once the caller's restack has put the layers on the map and they rendered
  }
  function removeVariable(name) {
    const vl = vlayers.get(name);
    if (!vl) return;
    for (const l of vl.layers) { state.map.getLayers().remove(l); l.dispose?.(); }
    vlayers.delete(name);
  }
  // Put the existing layers on the map in stacking order (hidden variables have none).
  function restack() {
    const layers = state.map.getLayers();
    for (const n of state.selected) for (const layer of vlayers.get(n)?.layers || []) layers.remove(layer);
    for (const n of state.selected) for (const layer of vlayers.get(n)?.layers || []) layers.push(layer);
  }

  // reference moveVariable: put `name` in the selection (before = the variable it goes in front of in the top-first
  // list; null = bottom) or take it out.
  function moveVariable(name, selected, before = null) {
    const i = state.selected.indexOf(name);
    if (!selected && i < 0) return;
    if (selected && before === name) return;
    if (i >= 0) state.selected.splice(i, 1);
    if (selected) {
      // top-first for the UI, state.selected is bottom-first for the map
      const order = [...state.selected].reverse();
      const index = before === null ? order.length : order.indexOf(before);
      order.splice(index < 0 ? order.length : index, 0, name);
      state.selected = order.reverse();
      if (i < 0) addVariable(name);
      restack();  // reuses existing layers (and their sources/styles) in the new stacking order
    } else {
      state.hidden.delete(name);
      removeVariable(name);
    }
    app.url.sync();
    bus.emit('selection:changed', [...state.selected]);
  }

  const layers = {
    basemaps: BASEMAPS,
    unionDims, opacityOf,
    group: (name) => vlayers.get(name),
    // Add a variable to the map. `before`: the selected variable it goes in front of (top-first list); default: on top.
    // Selecting an already selected variable without `before` does nothing.
    select(name, before) {
      if (!varOf(name)) return;
      if (state.selected.includes(name) && before === undefined) return;
      moveVariable(name, true, before === undefined ? (state.selected.filter((n) => n !== name).at(-1) || null) : before);
    },
    deselect: (name) => moveVariable(name, false),
    // reorder a selected variable (same `before` rule as select)
    move(name, before = null) { if (state.selected.includes(name)) moveVariable(name, true, before); },
    setVisible(name, on) {
      if (!state.selected.includes(name) || state.hidden.has(name) === !on) return;
      if (on) { state.hidden.delete(name); addVariable(name); restack(); }
      else { state.hidden.add(name); removeVariable(name); }
      bus.emit('visibility:changed', name);
      app.url.sync();
    },
    setDim(name, i) {
      state.dims[name] = i;
      app.url.sync();
      for (const [vn, vl] of vlayers) {
        if (!varOf(vn).dims.includes(name)) continue;
        for (const src of vl.sources) src.updateDimensions({[name]: i});
      }
      bus.emit('dims:changed', {dim: name, index: i});
    },
    setOpacity(name, x) {
      const v = clamp(Number(x), 0, 1);
      if (!Number.isFinite(v)) return;
      state.opacity[name] = v;
      for (const l of vlayers.get(name)?.layers || []) l.setOpacity(v);
      app.url.sync();
      bus.emit('opacity:changed', name);
    },
    setBasemap(id) {
      if (!BASEMAPS.some(([k]) => k === id)) return;
      state.basemap = id;
      for (const [k, l] of Object.entries(basemaps)) l.setVisible(k === id);
      landTop.setVisible(id === 'land');
      if (id === 'land') showLand();
      if (state.map) app.url.sync();  // before the map exists the app syncs once at the end of startup
      bus.emit('basemap:changed', id);
    },
    // Rebuilding the view in place is unreliable (a blank map), so reload with the new centre.
    setCenterLon(lon) {
      lon = normLon(+lon);
      if (!app.cfg.global || lon === state.centerLon) return;
      app.storage.set(`centerlon:${app.cfg.title}`, String(lon));
      bus.emit('centerlon:changed', lon);
      const url = new URL(location.href);
      url.searchParams.set('lon', String(lon));
      for (const k of ['x', 'y', 'z']) url.searchParams.delete(k);
      location.replace(url.href);
    },
    // Called once by the app after the config loaded and the URL was read: register the CRS, create the visible
    // variables' layers and the map.
    start() {
      const cfg = app.cfg;
      state.centerLon = cfg.global
        ? normLon(+(app.params.get('lon') ?? app.storage.get(`centerlon:${cfg.title}`) ?? 205)) : 0;
      // The CRS has to be registered before the GeoZarr source resolves it.
      if (cfg.crs.proj4) { proj4.defs(cfg.crs.name, cfg.crs.proj4); register(proj4); }
      layers.setBasemap(state.basemap);
      for (const name of state.selected) if (!state.hidden.has(name)) vlayers.set(name, makeLayers(varOf(name)));
      const dataLayers = state.selected.flatMap((n) => vlayers.get(n)?.layers || []);
      const lon = state.centerLon;
      let view;
      if (app.blank) {
        const b = cfg.bbox;
        const opts = viewOptions({projection: cfg.crs.code, center: [lon, 0], zoom: 2, showFullExtent: true,
          extent: [lon - 180, b[1], lon + 180, b[3]]}, cfg, lon, app.params);
        // A Promise<ViewOptions>: a View built here fails Map's instanceof check (separate +esm bundles).
        view = Promise.resolve(opts);
      } else {
        // With no visible layer, a temporary source supplies the view; it is disposed and never added to the map.
        const detached = !vlayers.size;
        const initial = vlayers.values().next().value || makeLayers(varOf(state.selected[0]) || cfg.variables[0]);
        view = getView(initial.sources[0], withLowerResolutions(1), withHigherResolutions(2), withExtentCenter(), withZoom(2))
          .then((o) => viewOptions(o, cfg, lon, app.params))
          .finally(() => { if (detached) for (const l of initial.layers) l.dispose?.(); });
        view.catch((e) => bus.emit('error', e));
      }
      state.map = new Map({target: app.mapEl, layers: [...Object.values(basemaps), landTop, ...dataLayers], view});
      state.map.on('moveend', app.url.sync);
      state.map.on('rendercomplete', () => { for (const n of vlayers.keys()) bus.emit('layer:rendered', n); });
      bus.on('style:changed', (name) => {
        for (const l of vlayers.get(name)?.layers || []) l.setStyle({color: app.style.colorExpr(name)});
      });
      for (const name of state.selected) if (vlayers.has(name)) autoOnAdd(name);  // visible variables only (D-53)
      bus.emit('map:ready', state.map);
    },
  };

  // lon/lat helpers for readouts: coordinates are in the view projection and, for global data, may sit on a world copy.
  app.geo = {
    // [lon, lat] (lon in [-180, 180)) of a view coordinate, or null when it cannot be transformed
    lonLat(coord) {
      const [x, y] = coord;
      const [lon, lat] = toLonLat([app.cfg.global ? normLon(x) : x, y], state.map.getView().getProjection());
      return Number.isFinite(lon) && Number.isFinite(lat) ? [normLon(lon), lat] : null;
    },
    // whether a view coordinate is inside the data extent
    inExtent(coord) {
      let [x, y] = coord;
      const b = app.cfg.bbox;
      if (app.cfg.global) x = normLon(x);  // map shows +-360 copies of the world
      return !!b && x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
    },
  };
  return layers;
}

// ---- core/frame.js
// Data of the current frame (loaded tiles of the current dimension slice): Auto and Min/Max ranges, pixel sampling.
// Ported from the reference: frameTiles, scanTile, frameValues, frameExtent, minMaxRange, symmetricCmap, autoRange,
// sampleLayer. The pure functions take a variable's layer group `vl` = {layers, sources} (see layers.js).
const AUTO_MAX = 200000;  // values collected for Auto at most

// Loaded tiles of the variable's current dimension slice (D-48), as [{tile, g}] with g the gutter. Same tile
// cache as sampleLayer (no store I/O): only tiles whose `key` is the source's current key (D-41) and only the
// highest loaded zoom among them, every world copy included. Shared by frameValues (Auto) and frameExtent (Min/Max).
function frameTiles(vl) {
  const tiles = [];
  let z = -1;
  if (vl) vl.layers.forEach((layer, i) => {
    const cache = layer.getRenderer?.()?.tileRepresentationCache;
    if (!cache) return;
    const key = vl.sources[i].getKey();
    cache.forEach((rep) => {
      const tile = rep && rep.loaded && rep.tile;
      if (!tile || tile.key !== key || tile.getState() !== 2) return;  // 2 = TileState.LOADED
      const tz = tile.tileCoord[0];
      if (tz > z) { z = tz; tiles.length = 0; }
      if (tz === z) tiles.push({tile, g: rep.gutter || 0});
    });
  });
  return tiles;
}

// Calls fn(value) for the band-0 value of every `step`-th pixel (rows and columns) of one tile. Gutter and alpha as
// in sampleLayer (last band when there are several, 0 = nodata); non-finite values are skipped.
function scanTile({tile, g}, step, fn) {
  const data = tile.getData();
  const [w, h] = tile.getSize();
  if (!data || !data.length) return;
  const bands = Math.max(1, Math.round(data.length / ((w + 2 * g) * (h + 2 * g))));
  for (let row = 0; row < h; row += step) {
    for (let col = 0; col < w; col += step) {
      const px = ((row + g) * (w + 2 * g) + col + g) * bands;
      if (bands > 1 && !data[px + bands - 1]) continue;
      const val = data[px];
      if (Number.isFinite(val)) fn(val);
    }
  }
}

// Band-0 values of the loaded tiles of the current slice (frameTiles), for the Auto range. A common 2-D stride keeps
// at most ~AUTO_MAX values. Returns a Float64Array (empty when nothing is loaded).
function frameValues(vl) {
  const tiles = frameTiles(vl);
  let total = 0;
  for (const t of tiles) { const [w, h] = t.tile.getSize(); total += w * h; }
  const step = Math.max(1, Math.ceil(Math.sqrt(total / AUTO_MAX)));
  const out = new Float64Array(tiles.reduce((n, t) => {
    const [w, h] = t.tile.getSize();
    return n + Math.ceil(w / step) * Math.ceil(h / step);
  }, 0));
  let n = 0;
  for (const t of tiles) scanTile(t, step, (val) => { out[n++] = val; });
  return out.subarray(0, n);
}

// {min, max} of every pixel of the loaded tiles of the current slice (D-51), or null without finite values (with
// `positive`, without values > 0: the log scale of D-54 needs a positive range).
// Streams over the tiles with stride 1 (no subsampling, no copy), so the true extremes are found.
function frameExtent(vl, positive) {
  let min = Infinity, max = -Infinity;
  for (const t of frameTiles(vl)) {
    scanTile(t, 1, (val) => {
      if (positive && !(val > 0)) return;
      if (val < min) min = val;
      if (val > max) max = val;
    });
  }
  return min <= max ? {min, max} : null;
}

// {vmin, vmax} for the Min/Max button (D-51); vmax = vmin + 1 when the data is constant; null without finite limits.
function minMaxRange(min, max) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  return {vmin: min, vmax: max > min ? max : min + 1};
}

// Colourmap for a symmetric Auto range (D-49): a diverging map (any group labelled "... Diverging", so matplotlib,
// cmocean and cmasher) is kept with its Reversed state, every other map becomes RdBu reversed (red = high).
function symmetricCmap(cmap, reversed, groups) {
  const diverging = groups.some((g) => /Diverging$/.test(g.label) && g.names.includes(cmap));
  return diverging ? {cmap, reversed} : {cmap: 'RdBu', reversed: true};
}

// {vmin, vmax} from the 2nd and 98th percentiles (linear interpolation, as the default limits, D-43), or null
// without finite values. Mirrored to [-m, m] when lo < 0 < hi and the smaller magnitude is at least half the
// larger (`symmetric: true`); vmax = vmin + 1 when the range is empty. With `positive` (log scale, D-54) only
// values > 0 count, so the range stays positive.
function autoRange(values, positive) {
  const v = Float64Array.from(Array.from(values).filter((x) => Number.isFinite(x) && (!positive || x > 0))).sort();
  if (!v.length) return null;
  const pct = (p) => {
    const x = p * (v.length - 1), i = Math.floor(x);
    return i + 1 < v.length ? v[i] + (v[i + 1] - v[i]) * (x - i) : v[i];
  };
  let lo = pct(0.02), hi = pct(0.98);
  const a = Math.abs(lo), b = Math.abs(hi);
  const symmetric = lo < 0 && hi > 0 && Math.min(a, b) / Math.max(a, b) >= 0.5;
  if (symmetric) { hi = Math.max(a, b); lo = -hi; }
  if (!(hi > lo)) hi = lo + 1;
  return {vmin: lo, vmax: hi, symmetric};
}

// Raw band-0 value under `coordinate`, read from the layer renderer's loaded-tile cache (null if none/NaN).
// WebGLTileLayer.getData is not usable: with the jsdelivr +esm bundles the GeoZarr source's DataTile
// is not the DataTile class WebGLTile checks with instanceof, and the source itself caches no tiles.
// Edge tiles are smaller than the grid tile size, so pixels are addressed by resolution, not extent.
// The renderer keeps tiles of earlier dimension slices (the source key changes with `dimensions`), so only
// tiles whose `key` is the source's current key are used; until those are loaded there is no value (D-41).
function sampleLayer(layer, source, coordinate) {
  const cache = layer.getRenderer?.()?.tileRepresentationCache;
  const grid = source.getTileGrid();
  if (!cache || !grid) return null;
  const [cx, cy] = coordinate;
  let best = null;
  cache.forEach((rep) => {
    const tile = rep && rep.loaded && rep.tile;
    if (!tile || tile.key !== source.getKey() || tile.getState() !== 2) return;  // 2 = TileState.LOADED
    const tc = tile.tileCoord;
    if (best && best.z >= tc[0]) return;
    const ext = grid.getTileCoordExtent(tc);
    const res = grid.getResolution(tc[0]);
    const col = Math.floor((cx - ext[0]) / res), row = Math.floor((ext[3] - cy) / res);
    const [w, h] = tile.getSize();
    if (col < 0 || row < 0 || col >= w || row >= h) return;
    best = {z: tc[0], tile, col, row, g: rep.gutter || 0, w, h};
  });
  if (!best) return null;
  const {tile, col, row, g, w, h} = best;
  const data = tile.getData();
  if (!data || !data.length) return null;
  const bands = Math.max(1, Math.round(data.length / ((w + 2 * g) * (h + 2 * g))));
  const px = ((row + g) * (w + 2 * g) + col + g) * bands;
  if (bands > 1 && !data[px + bands - 1]) return null;  // last band is alpha; 0 = nodata
  const val = data[px];
  return Number.isFinite(val) ? val : null;
}

// app.frame: the helpers above bound to the app's layers and styles.
function createFrame(app) {
  const group = (name) => app.layers.group(name);
  const frame = {
    // band-0 values of the current frame (Float64Array, subsampled to ~AUTO_MAX)
    values: (name) => frameValues(group(name)),
    // {min, max} of the current frame or null; `positive` for log scales
    extent: (name, positive) => frameExtent(group(name), positive),
    // Value of every visible variable under a view-projection coordinate, top-most first: [{name, value}].
    // Global data has three world copies (0, -360, +360); the first finite value wins.
    sample(coordinate) {
      const out = [];
      for (const name of [...app.state.selected].reverse()) {
        const vl = group(name);
        if (!vl || app.state.hidden.has(name)) continue;
        let value = null;
        for (let i = 0; i < vl.layers.length; i++) {
          try { value = sampleLayer(vl.layers[i], vl.sources[i], coordinate); } catch (_) { value = null; }  // not rendered yet
          if (value !== null) break;
        }
        if (value !== null) out.push({name, value});
      }
      return out;
    },
    // Auto range for the variable's current frame, applied through app.style (D-48/D-49): 2-98 % percentiles, and a
    // diverging colourmap when the signs balance. `auto: true` is the automatic run after a layer is added (D-53),
    // which neither stores nor marks the range as explicit. Returns false (style kept) when no data is loaded.
    autoStyle(name, {auto = false} = {}) {
      const st = app.style.get(name);
      const r = st && autoRange(frame.values(name), !!st.log);
      if (!r) return false;
      const patch = {vmin: r.vmin, vmax: r.vmax};
      if (r.symmetric) Object.assign(patch, symmetricCmap(st.cmap, st.reversed, app.style.groups));
      return app.style.set(name, patch, {explicit: !auto, persist: !auto});
    },
    // Min/Max of the current frame (D-51); the colourmap is untouched. Returns false without loaded data.
    minMaxStyle(name) {
      const st = app.style.get(name);
      const e = st && frame.extent(name, !!st.log);
      const r = e && minMaxRange(e.min, e.max);
      return !!r && app.style.set(name, r);
    },
  };
  return frame;
}

// ---- core/url.js
// Shareable URL: a registry of {read(params), write(params)} entries. The core registers store, var/hidden, dims, style,
// basemap and view (lon, x, y, z); components register their own with app.url.register. Ported from the reference: dimToParam,
// dimFromParam, styleParams, applyUrlState, syncUrl. URL values win over remembered (localStorage) ones, so a shared
// link reproduces the view.

const dimToParam = (d, i, dims) => {
  const iso = dims[d].iso;
  return iso && iso[i] ? iso[i].replace(/T00:00:00(\.0+)?Z$/, '') : String(i);
};
function dimFromParam(d, raw, dims) {
  const {iso, size} = dims[d];
  if (iso && /^\d{4}-/.test(raw)) {  // a date prefix; a bare number is an index (the reference matched it as a prefix too)
    const i = iso.findIndex((t) => t && t.startsWith(raw));
    if (i >= 0) return i;
  }
  const i = Number(raw);
  return Number.isInteger(i) && i >= 0 && i < size ? i : null;
}

// Style/opacity parameters: plain names for one selected variable, `<param>.<var>` for several.
// getStyle(name) -> style, opacityOf(name) -> number. Returns [[key, value], ...].
function styleParamsOf(vars, getStyle, opacityOf) {
  const multi = vars.length > 1, out = [];
  for (const v of vars) {
    const key = (k) => (multi ? `${k}.${v.name}` : k);
    if (!v.categorical) {
      const st = getStyle(v.name);
      out.push([key('cmap'), st.cmap]);
      if (st.reversed) out.push([key('rev'), '1']);
      if (st.log) out.push([key('log'), '1']);
      out.push([key('vmin'), String(+Number(st.vmin).toPrecision(6))]);
      out.push([key('vmax'), String(+Number(st.vmax).toPrecision(6))]);
    }
    if (opacityOf(v.name) !== 1) out.push([key('opacity'), String(opacityOf(v.name))]);
  }
  return out;
}

// On load a plain parameter applies to the first selected variable (idx 0), `<param>.<var>` to the named one.
// base: the variable's current style. Returns {style: full style or null when the URL changes nothing valid,
// explicit: whether the range/colourmap came from the URL (D-53), opacity: number or null}.
function readStyleParams(params, name, idx, base, cmaps) {
  const get = (k) => (params.has(`${k}.${name}`) ? params.get(`${k}.${name}`) : (idx === 0 && params.has(k) ? params.get(k) : null));
  const op = get('opacity');
  const opacity = op !== null && op !== '' && Number(op) >= 0 && Number(op) <= 1 ? Number(op) : null;
  const st = {...base};
  let changed = false;
  const cm = get('cmap');
  if (cm && cmaps[cm]) { st.cmap = cm; st.reversed = false; changed = true; }  // the URL always writes `rev` when set, so absent = off
  const rv = get('rev');
  if (rv !== null) { st.reversed = rv === '1'; changed = true; }
  const lg = get('log');
  if (lg !== null) { st.log = lg === '1'; changed = true; }
  const lo = get('vmin'), hi = get('vmax');
  if (lo !== null && lo !== '' && Number.isFinite(Number(lo))) { st.vmin = Number(lo); changed = true; }
  if (hi !== null && hi !== '' && Number.isFinite(Number(hi))) { st.vmax = Number(hi); changed = true; }
  if (!changed || !(st.vmin < st.vmax)) return {style: null, explicit: false, opacity};
  const explicit = ['cmap', 'rev', 'vmin', 'vmax'].some((k) => get(k) !== null && get(k) !== '');
  return {style: st, explicit, opacity};
}

// "/path?store=s3://b/x&var=a,b": URLSearchParams escapes / : , which are harmless and unreadable in a query.
const urlString = (pathname, q) =>
  `${pathname}?${q.toString().replace(/%2F/g, '/').replace(/%3A/g, ':').replace(/%2C/g, ',')}`;

// app.url: register(key, {read(params), write(params)}), read() (once, after the config loaded), sync() (debounced).
function createUrl(app) {
  const entries = [];
  let timer = null;
  return {
    register(key, entry) { entries.push([key, entry]); },
    read(params = app.params) {
      for (const [key, e] of entries) {
        try { e.read?.(params); } catch (err) { console.error(`url "${key}" read failed`, err); }
      }
    },
    sync() {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const q = new URLSearchParams();
        for (const [key, e] of entries) {
          try { e.write?.(q); } catch (err) { console.error(`url "${key}" write failed`, err); }
        }
        try { history.replaceState(null, '', urlString(location.pathname, q)); } catch (_) { /* sandboxed frame */ }
      }, 150);
    },
  };
}

// The core's own URL entries: store, selection (var, hidden), dims, style (+ opacity), basemap, view (lon, x, y, z).
function registerCoreUrl(app) {
  const {state} = app;
  app.url.register('store', {
    write(q) {
      const cfg = app.cfg;
      if (app.blank) { if (app.spec.store) q.set('store', ''); return; }  // `?store=` keeps a page with a default store blank
      q.set('store', cfg.source);
      if (cfg.endpoint && /^s3:\/\//i.test(cfg.source)) q.set('endpoint', cfg.endpoint);
    },
  });
  app.url.register('selection', {
    read(p) {
      if (app.blank) return;
      const names = [];
      for (const n of (p.get('var') || '').split(',')) if (app.varOf(n) && !names.includes(n)) names.push(n);
      if (names.length || p.get('var') === '') state.selected = names;
      for (const name of (p.get('hidden') || '').split(',')) if (state.selected.includes(name)) state.hidden.add(name);
    },
    write(q) {
      if (app.blank) return;
      q.set('var', state.selected.join(','));
      const hidden = state.selected.filter((n) => state.hidden.has(n));
      if (hidden.length) q.set('hidden', hidden.join(','));
    },
  });
  app.url.register('dims', {
    read(p) {
      if (app.blank) return;
      for (const d of Object.keys(app.cfg.dims)) {
        const raw = p.get(d);
        if (raw === null) continue;
        const i = dimFromParam(d, raw, app.cfg.dims);
        if (i !== null) state.dims[d] = i;
      }
    },
    write(q) {
      if (app.blank) return;
      for (const d of app.layers.unionDims()) q.set(d, dimToParam(d, state.dims[d] || 0, app.cfg.dims));
    },
  });
  app.url.register('style', {
    read(p) {
      if (app.blank) return;
      state.selected.forEach((name, idx) => {
        const v = app.varOf(name);
        const r = readStyleParams(p, name, idx, app.style.get(name) || {}, app.style.cmaps);
        if (r.opacity !== null) state.opacity[name] = r.opacity;
        if (!v.categorical && r.style) app.style.set(name, r.style, {explicit: r.explicit, persist: false});
      });
    },
    write(q) {
      if (app.blank) return;
      const vars = state.selected.map((n) => app.varOf(n));
      for (const [k, val] of styleParamsOf(vars, (n) => app.style.get(n), (n) => app.layers.opacityOf(n))) q.set(k, val);
    },
  });
  app.url.register('basemap', {
    read(p) { if (app.layers.basemaps.some(([k]) => k === p.get('basemap'))) state.basemap = p.get('basemap'); },
    write(q) { q.set('basemap', state.basemap); },
  });
  app.url.register('view', {
    write(q) {
      if (app.cfg.global) q.set('lon', String(state.centerLon));
      const view = state.map && state.map.getView();
      const c = view && view.getCenter && view.getCenter();
      if (c && Number.isFinite(view.getZoom())) {
        q.set('x', c[0].toPrecision(6)); q.set('y', c[1].toPrecision(6));
        q.set('z', view.getZoom().toFixed(2));
      }
    },
  });
}

// ---- core/app.js
// Builds `app`, runs the component inits, loads the store and boots the map. The assembler puts this file after the
// other core files and before the components, and ends the script with `bootViewer();`.

// Components register themselves (one call each); bootViewer inits them in spec order.
const COMPONENTS = {};
function registerComponent(def) { COMPONENTS[def.name] = def; }

// theme = auto | light | dark: data-theme on <html> (theme.css: auto follows prefers-color-scheme)
function applyTheme(theme) {
  document.documentElement.dataset.theme = ['light', 'dark'].includes(theme) ? theme : 'auto';
}

// Core owns the sidebar collapse toggle (a layout option): button + vertical title ("spine") when collapsed.
function initLayout(app) {
  const root = $('gv-app');
  const side = root.querySelector('.gv-sidebar');
  if (!side || !app.spec.layout.sidebar_collapsible) return;
  root.setAttribute('data-collapsible', '');
  const btn = h('button', 'gv-sbtoggle');
  btn.type = 'button';
  btn.innerHTML = ICON.prev;
  const spine = h('button', 'gv-spine');
  spine.type = 'button';
  spine.append(h('span'));
  side.prepend(btn, spine);
  const set = (collapsed) => {
    root.classList.toggle('collapsed', collapsed);
    btn.setAttribute('aria-expanded', String(!collapsed));
    btn.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
    app.storage.set('sidebar-collapsed', collapsed ? '1' : '0');
    window.dispatchEvent(new Event('resize'));  // the map resizes with its column
  };
  btn.addEventListener('click', () => set(!root.classList.contains('collapsed')));
  spine.addEventListener('click', () => set(false));
  spine.firstChild.textContent = app.spec.title || 'GeoZarr viewer';
  set(app.storage.get('sidebar-collapsed') === '1');
}

// Core owns the sidebar resize handle (layout option sidebar_resizable): drag the inner edge, the width is stored per
// browser; double-click resets it to the spec width. The handle is hidden while the sidebar is collapsed.
const SB_MIN = 120, SB_MAX = 900;
function initSidebarResize(app) {
  const root = $('gv-app');
  const side = root.querySelector('.gv-sidebar');
  if (!side || !app.spec.layout.sidebar_resizable) return;
  const right = root.dataset.sidebar === 'right';
  const clampW = (w) => Math.round(Math.max(SB_MIN, Math.min(SB_MAX, window.innerWidth - 200, w)));
  const setW = (w) => { root.style.setProperty('--gv-sb-width', w + 'px'); window.dispatchEvent(new Event('resize')); };
  const handle = h('div', 'gv-sbresize');
  handle.setAttribute('role', 'separator');
  handle.setAttribute('aria-orientation', 'vertical');
  handle.title = 'Drag to resize, double-click to reset';
  side.append(handle);
  let drag = null;
  handle.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    drag = {x: e.clientX, w: side.getBoundingClientRect().width};
    handle.setPointerCapture(e.pointerId);
    root.classList.add('resizing');
  });
  handle.addEventListener('pointermove', (e) => {
    if (!drag) return;
    setW(clampW(drag.w + (e.clientX - drag.x) * (right ? -1 : 1)));
  });
  const end = () => {
    if (!drag) return;
    drag = null;
    root.classList.remove('resizing');
    app.storage.set('sidebar-width', String(Math.round(side.getBoundingClientRect().width)));
  };
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);
  handle.addEventListener('dblclick', () => { app.storage.remove('sidebar-width'); setW(app.spec.layout.sidebar_width); });
  const stored = +app.storage.get('sidebar-width');
  if (stored > 0) setW(clampW(stored));
}

// Collapsible groups (spec `group`): the assembler emits <section class="gv-group" data-gv-group> with the body
// as its only child; this adds the header button (icon, title, chevron). The state is stored per group name (D-46).
function initGroups(app) {
  for (const section of document.querySelectorAll('.gv-group')) {
    const name = section.dataset.gvGroup;
    const info = (app.spec.groups || {})[name] || {};
    const body = section.firstChild;
    const head = h('button', 'gv-group-head');
    head.type = 'button';
    head.setAttribute('aria-controls', body.id);
    if (info.icon !== 'none') { const ic = h('span', 'gv-group-icon'); ic.innerHTML = ICON.gear; head.append(ic); }
    head.append(h('span', 'gv-group-title', name));
    const chev = h('span', 'gv-group-chev');
    chev.innerHTML = ICON.chevDown;
    head.append(chev);
    const set = (collapsed, save) => {
      body.hidden = collapsed;
      section.classList.toggle('collapsed', collapsed);
      head.setAttribute('aria-expanded', String(!collapsed));
      if (save) app.storage.set('group:' + name, collapsed ? '1' : '0');
    };
    head.addEventListener('click', () => set(!body.hidden, true));
    section.prepend(head);
    const stored = app.storage.get('group:' + name);
    set(stored === null ? !!info.collapsed : stored === '1', false);
  }
}

// Errors (store failed to open, tile errors, component errors): one dismissible toast on the map.
function initErrors(app) {
  const wrap = document.querySelector('.gv-mapwrap');
  const box = h('div', 'gv-toast');
  box.setAttribute('role', 'alert');
  box.hidden = true;
  box.title = 'Click to dismiss';
  box.addEventListener('click', () => { box.hidden = true; });
  wrap.append(box);
  app.bus.on('error', (e) => {
    console.error(e);
    const err = e && e.error ? e.error : e;  // an ErrorEvent carries the error
    let hint = err && err.hint || '';
    if (location.protocol === 'file:') hint = (hint ? hint + ' ' : '') + 'This page was opened from disk; serve it with `scripts/serve.py` instead.';
    box.innerHTML = `<div class="msg">${codeify(String(err && err.message || err))}</div>` + (hint ? `<div class="hint">${codeify(hint)}</div>` : '');
    box.hidden = false;
  });
}

async function bootViewer() {
  const spec = JSON.parse(document.getElementById('spec').textContent);
  applyTheme(spec.theme);
  const app = {
    spec, params: new URLSearchParams(location.search),
    cfg: blankConfig(), blank: true, startupError: null,
    state: {selected: [], hidden: new Set(), dims: {}, roles: {}, opacity: {}, centerLon: 0, map: null,
      labelMode: 'name', basemap: 'dark'},
    bus: createBus(), storage: makeStorage('gv:'),
    mapEl: $('gv-map'),
    slot: (name) => document.querySelector(`[data-slot="${name}"]`),
    varOf: (name) => app.cfg.variables.find((v) => v.name === name),
  };
  // D-44: variables are shown by array name or, when it has one, by long name. Not in the URL.
  app.state.labelMode = app.storage.get('label-mode') === 'long' ? 'long' : 'name';
  app.label = (v) => {
    if (typeof v === 'string') v = app.varOf(v) || {name: v};
    return app.state.labelMode === 'long' && v.long_name?.trim() ? (v.display_name || v.long_name) : v.name;
  };
  app.setLabelMode = (mode) => {
    app.state.labelMode = mode === 'long' ? 'long' : 'name';
    app.storage.set('label-mode', app.state.labelMode);
    app.bus.emit('labels:changed', app.state.labelMode);
  };
  // Open another store: validate it, then reload the page on it (a fresh URL drops the old settings).
  app.openStore = async (src, endpoint = '') => {
    await openStore(src, endpoint, spec.relay);
    const url = new URL(location.pathname, location.href);
    url.searchParams.set('store', src);
    if (endpoint) url.searchParams.set('endpoint', endpoint);
    location.assign(url.href);
  };
  app.url = createUrl(app);
  app.style = createStyle(app);
  app.layers = createLayers(app);
  app.frame = createFrame(app);
  registerCoreUrl(app);
  window.gvApp = app;  // for debugging and embedding scripts

  noHeadSuffixReads(OlFetchStore);  // D-56
  initLayout(app);
  initSidebarResize(app);
  initGroups(app);
  initErrors(app);

  // components first: they subscribe to config:loaded / map:ready, which fire below
  for (const c of spec.components) {
    const def = COMPONENTS[c.name];
    const root = document.querySelector(`[data-gv-component="${c.name}"]`);
    if (!def || !root) { app.bus.emit('error', new Error(`component "${c.name}" is not available in this page`)); continue; }
    try { def.init(app, root, c.options || {}); } catch (e) { app.bus.emit('error', e); }
  }

  // store: ?store= wins over the spec; `?store=` (empty) is the blank viewer even on a page with a default store
  const wanted = app.params.get('store');
  const src = wanted !== null ? wanted : (spec.store || '');
  const endpoint = wanted !== null ? (app.params.get('endpoint') || '') : (spec.endpoint || '');
  let cfg = blankConfig();
  if (src) {
    try { cfg = await openStore(src, endpoint, spec.relay); } catch (e) { app.startupError = e; app.bus.emit('error', e); }
  }
  app.cfg = cfg;
  app.blank = !cfg.variables.length;
  document.title = spec.title || (app.blank ? 'GeoZarr viewer' : `${cfg.title} | GeoZarr viewer`);
  if (!app.blank) app.state.selected = [cfg.variables[0].name];
  app.url.read();
  app.bus.emit('config:loaded', cfg);
  app.layers.start();
  app.url.sync();
}

// ---- component: info-panel
{
// Dataset information: an info button and a panel. Reference: buildInfo (D-32, D-55; attribute values arrive formatted in cfg.attrs). Rows: title, store,
// CRS, bbox, variables, dims, levels, then the global attributes. In the sidebar the panel opens inline under the
// button; elsewhere it is a popover beside the button, kept inside the viewport.

// Sections [{title, rows: [[key, value, id?]]}] for a dataset config; `label` names a variable (array or long name).
function infoSections(cfg, blank, label) {
  const rows = [];
  if (blank) rows.push(['Store', 'none loaded; type a path or URL in the store picker and press Enter']);
  else {
    if (cfg.title) rows.push(['Title', cfg.title]);
    rows.push(['Store', cfg.source || cfg.title]);
  }
  rows.push(['CRS', cfg.crs.name]);
  if (cfg.bbox) rows.push(['Bbox', cfg.bbox.map((x) => +Number(x).toPrecision(7)).join(', ')]);
  if (!blank) {
    rows.push([`Variables (${cfg.variables.length})`, cfg.variables.map(label).join(', '), 'info-vars']);
    for (const [d, info] of Object.entries(cfg.dims || {})) rows.push([`Dimension ${d}`, `${info.size} steps`]);
    if (cfg.levels) rows.push(['Levels', cfg.levels]);
  }
  const sections = [{title: 'Metadata', rows}];
  const attrs = Object.entries(cfg.attrs || {});
  if (attrs.length) sections.push({title: 'Global attributes', rows: attrs});
  return sections;
}

registerComponent({
  name: 'info-panel',
  init(app, root, opts) {
    const inline = root.closest('[data-slot]')?.dataset.slot === 'sidebar';
    root.dataset.mode = inline ? 'inline' : 'popover';
    const btn = h('button', 'gv-iconbtn ip-btn');
    btn.type = 'button';
    btn.innerHTML = ICON.info;
    btn.setAttribute('aria-label', 'Dataset information');
    btn.setAttribute('aria-expanded', 'false');
    btn.title = 'Dataset information';
    const panel = h('div', 'ip-panel');
    panel.hidden = true;
    root.append(btn, panel);

    function place() {  // popover: below the button, right edge aligned, kept in the viewport
      if (inline || panel.hidden) return;
      const at = btn.getBoundingClientRect();
      const left = Math.max(8, Math.min(at.right - panel.offsetWidth, innerWidth - panel.offsetWidth - 8));
      panel.style.left = `${left}px`;
      panel.style.top = `${Math.max(8, Math.min(at.bottom + 4, innerHeight - panel.offsetHeight - 8))}px`;
    }
    const setOpen = (open) => {
      panel.hidden = !open;
      btn.setAttribute('aria-expanded', String(open));
      place();
    };
    function fill() {
      panel.replaceChildren();
      for (const {title, rows} of infoSections(app.cfg, app.blank, (v) => app.label(v))) {
        panel.append(h('h2', '', title));
        panel.append(kvTable(rows));
      }
      place();
    }
    btn.addEventListener('click', () => setOpen(panel.hidden));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !panel.hidden) setOpen(false); });
    addEventListener('resize', place);
    app.bus.on('config:loaded', () => { fill(); if (opts.open) setOpen(true); });
    app.bus.on('labels:changed', () => {  // relabel the variables row in place (D-44)
      const td = panel.querySelector('td[data-id="info-vars"]');
      if (td) td.textContent = app.cfg.variables.map((v) => app.label(v)).join(', ');
    });
  },
});
}

// ---- component: variable-cards
{
// Variable cards. Reference: buildCards, refreshCards, initCardDragging, selectedDropBefore, toggleVar (D-36/D-37/D-38),
// labelOf (D-44, here app.label). Legend, colour and opacity controls are not here: style components fill the card's
// settings container (event card:settings); var-attrs opens on varinfo:open (D-55).

// Card names by section: selected top-first (app.state.selected is bottom-first), the others in store order.
function cardOrder(variables, selected) {
  const sel = new Set(selected);
  return {selected: [...selected].reverse(), other: variables.map((v) => v.name).filter((n) => !sel.has(n))};
}

// Reference selectedDropBefore: the first selected card (top-first `names`) whose vertical midpoint is below `y`,
// ignoring the dragged one; null = dropped below all of them (= the bottom layer).
function dropBefore(names, rectOf, y, dragged) {
  return names.find((n) => {
    if (n === dragged) return false;
    const r = rectOf(n);
    return y < r.top + r.height / 2;
  }) || null;
}

// "unit · dim · dim" metadata line
function metaLine(v) {
  return [unitsOf(v), ...(v.dims || [])].filter(Boolean).join(' · ');
}

registerComponent({
  name: 'variable-cards',
  init(app, root, opts) {
    const sections = opts.sections !== false, drag = opts.drag !== false, showOther = opts.show_other !== false;
    const OTHER_KEY = 'other-collapsed';  // reference OTHER_KEY
    const hasAttrs = app.spec.components.some((c) => c.name === 'var-attrs');
    const cards = new globalThis.Map();  // name -> {el, title, longname, info, gear, visibility, remove, handle}
    let otherCollapsed = app.storage.get(OTHER_KEY) === '1';
    let dragged = null;
    const open = new Set();  // cards whose settings are expanded

    const selBox = h('div', 'vc-list'), otherBox = h('div', 'vc-list');
    const selHead = h('h2', 'vc-head'), selSec = h('section', 'vc-sec'), otherSec = h('section', 'vc-sec');
    const otherToggle = h('button', 'vc-head vc-toggle');
    otherToggle.type = 'button';
    otherToggle.innerHTML = `<svg class="chev" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    const otherLabel = h('span');
    otherToggle.append(otherLabel);
    selSec.append(selHead, selBox);
    otherSec.append(otherToggle, otherBox);
    const hint = h('div', 'vc-hint', 'Click to select. Use the eye to show or hide, and × to remove.' +
      `${drag ? ' Drag handles to reorder.' : ''} The top card is drawn on top.`);
    hint.hidden = true;
    root.append(hint, ...(sections ? [selSec, otherSec] : [selBox]));
    selSec.hidden = otherSec.hidden = true;

    // ---- one card (reference buildCards)
    function buildCard(v) {
      const name = v.name;
      const el = h('div', 'vc-card');
      el.tabIndex = 0;
      el.setAttribute('role', 'button');
      const top = h('div', 'vc-top');
      const title = h('div', 'vc-name');
      let handle = null;
      if (drag) {
        handle = h('span', 'gv-iconbtn vc-handle', '⠿');
        handle.setAttribute('role', 'img');
        handle.draggable = true;
        handle.title = 'Drag to reorder or move between Selected and Other';
        handle.addEventListener('click', (e) => e.stopPropagation());
        handle.addEventListener('dragstart', (e) => {
          dragged = name;
          e.dataTransfer.effectAllowed = 'move';
          e.dataTransfer.setData('text/plain', name);
          e.dataTransfer.setDragImage(el, 20, 20);
          el.classList.add('dragging');
        });
        handle.addEventListener('dragend', () => { dragged = null; el.classList.remove('dragging'); clearFeedback(); });
        top.append(handle);
      }
      top.append(title);
      const button = (cls, html, onclick) => {
        const b = h('button', `gv-iconbtn ${cls}`);
        b.type = 'button';
        if (html) b.innerHTML = html;
        b.addEventListener('click', (e) => { e.stopPropagation(); onclick(); });
        return b;
      };
      const info = button('vc-info', ICON.info, () => app.bus.emit('varinfo:open', {name, anchor: info}));
      info.title = 'Attributes';
      info.setAttribute('aria-expanded', 'false');
      info.hidden = !hasAttrs;
      const gear = button('vc-gear', ICON.gear, () => {
        if (open.has(name)) open.delete(name); else open.add(name);
        refresh();
      });
      gear.setAttribute('aria-label', 'Layer options');
      gear.setAttribute('aria-expanded', 'false');
      const visibility = button('vc-visibility', '', () => app.layers.setVisible(name, app.state.hidden.has(name)));
      const remove = button('vc-remove', '×', () => app.layers.deselect(name));
      remove.title = 'Remove from Selected';
      top.append(info, visibility, gear, remove);
      el.append(top);
      const meta = h('div', 'vc-meta', metaLine(v));
      meta.hidden = !meta.textContent;
      const longname = h('div', 'vc-longname');
      el.append(meta, longname);
      // settings container: style components fill it on card:settings; the gear is shown only when it has content
      const settings = h('div', 'vc-settings');
      settings.addEventListener('click', (e) => e.stopPropagation());
      el.append(settings);
      const sync = () => { gear.hidden = !settings.hasChildNodes(); };
      new MutationObserver(sync).observe(settings, {childList: true});
      gear.hidden = true;
      app.bus.emit('card:settings', {name, el: settings});
      sync();
      el.addEventListener('click', () => app.layers.select(name));  // a selected card is left as it is
      el.addEventListener('keydown', (e) => {
        if (e.target === el && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); app.layers.select(name); }
      });
      cards.set(name, {el, title, longname, info, gear, visibility, remove, handle});
      return el;
    }

    // ---- drag and drop into the sections (reference initCardDragging)
    function clearFeedback() {
      selSec.classList.remove('drop-target');
      otherSec.classList.remove('drop-target');
      for (const c of cards.values()) c.el.classList.remove('drop-before');
    }
    const beforeAt = (y) => dropBefore(cardOrder(app.cfg.variables, app.state.selected).selected,
      (n) => cards.get(n).el.getBoundingClientRect(), y, dragged);
    function dropTarget(section, selected) {
      section.addEventListener('dragover', (e) => {
        if (!dragged) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        clearFeedback();
        section.classList.add('drop-target');
        const before = selected ? beforeAt(e.clientY) : null;
        if (before) cards.get(before).el.classList.add('drop-before');
      });
      section.addEventListener('dragleave', (e) => { if (!section.contains(e.relatedTarget)) clearFeedback(); });
      section.addEventListener('drop', (e) => {
        if (!dragged) return;
        e.preventDefault();
        const name = dragged, before = selected ? beforeAt(e.clientY) : null;
        clearFeedback();
        if (!selected) app.layers.deselect(name);
        else if (app.state.selected.includes(name)) app.layers.move(name, before);
        else app.layers.select(name, before);
      });
    }
    if (drag && sections) { dropTarget(selSec, true); dropTarget(otherSec, false); }

    // ---- state -> DOM (reference refreshCards)
    function refresh() {
      const order = cardOrder(app.cfg.variables, app.state.selected);
      const place = (box, names) => box.append(...names.filter((n) => cards.has(n)).map((n) => cards.get(n).el));
      if (sections) {
        place(selBox, order.selected);
        place(otherBox, order.other);
        selSec.hidden = !cards.size;
        otherSec.hidden = !cards.size || !showOther;  // kept even when empty: it is a drop target
        selHead.textContent = `Selected (${order.selected.length})`;
        otherLabel.textContent = `Other (${order.other.length})`;
        otherToggle.setAttribute('aria-expanded', String(!otherCollapsed));
        otherBox.hidden = otherCollapsed;
      } else {
        place(selBox, [...order.selected, ...order.other]);  // flat list: selected first, top-first
      }
      for (const [name, c] of cards) {
        const active = app.state.selected.includes(name), visible = active && !app.state.hidden.has(name);
        const isOpen = active && open.has(name);
        const v = app.varOf(name);
        c.el.hidden = !active && !showOther;
        c.title.textContent = app.label(v);
        const other = app.state.labelMode === 'long' ? v.name : (v.long_name?.trim() ? (v.display_name || v.long_name) : '');
        c.longname.textContent = other;  // the other name, never a duplicate of the title (D-38/D-44)
        c.longname.hidden = !other;
        c.el.classList.toggle('selected', active);
        c.el.classList.toggle('active', visible);
        c.el.classList.toggle('open', isOpen);
        c.el.setAttribute('aria-pressed', String(active));
        c.visibility.innerHTML = app.state.hidden.has(name) ? ICON.eyeOff : ICON.eye;
        const action = `${visible ? 'Hide' : 'Show'} ${app.label(v)}`;
        c.visibility.title = action;
        c.visibility.setAttribute('aria-label', action);
        c.visibility.setAttribute('aria-pressed', String(visible));
        c.gear.setAttribute('aria-expanded', String(isOpen));
        c.info.setAttribute('aria-label', `Attributes of ${app.label(v)}`);
        c.remove.setAttribute('aria-label', `Remove ${app.label(v)} from Selected`);
        c.handle?.setAttribute('aria-label', `Drag ${app.label(v)}`);
      }
      hint.hidden = cards.size < 2;
    }

    otherToggle.addEventListener('click', () => {
      otherCollapsed = !otherCollapsed;
      app.storage.set(OTHER_KEY, otherCollapsed ? '1' : '0');
      refresh();
    });
    app.bus.on('config:loaded', (cfg) => {
      for (const v of cfg.variables) selBox.append(buildCard(v));
      refresh();
    });
    for (const ev of ['selection:changed', 'visibility:changed', 'labels:changed']) app.bus.on(ev, refresh);
    // var-attrs tells which card's popover is open (aria-expanded of the info icon)
    app.bus.on('varinfo:state', ({name}) => {
      for (const [n, c] of cards) c.info.setAttribute('aria-expanded', String(n === name));
    });
  },
});
}

// ---- component: label-mode
{
// Label by: Array name / Long name (D-44). Reference: initLabelMode. The choice is app.setLabelMode (kept in storage,
// not in the URL); every component that shows a label listens to `labels:changed`.

const LABEL_MODES = [['name', 'Array name'], ['long', 'Long name']];

registerComponent({
  name: 'label-mode',
  init(app, root) {
    root.classList.add('gv-panel', 'gv-row');
    root.hidden = true;  // no variables yet (and none in the blank viewer)
    const sel = h('select');
    sel.id = 'gv-label-mode';
    sel.setAttribute('aria-label', 'Label by');
    for (const [value, text] of LABEL_MODES) sel.add(new Option(text, value));
    const lab = h('label', '', 'Label by');
    lab.htmlFor = sel.id;
    root.append(lab, sel);
    sel.value = app.state.labelMode;
    sel.addEventListener('change', () => app.setLabelMode(sel.value));
    app.bus.on('labels:changed', (mode) => { sel.value = mode; });
    app.bus.on('config:loaded', (cfg) => { root.hidden = !cfg.variables.length; sel.value = app.state.labelMode; });
  },
});
}

// ---- component: dim-roles
{
// Role select per dim: Horizontal (timeline), Vertical (dim-sliders) or Off (no control, the index stays applied).
// Reference: buildDimRoles, setRole, syncRoles, ROLES, the `slider.<dim>` URL entries (D-40, D-28). The model is
// app.state.roles; timeline and dim-sliders read it too and work without this component (defaults: first dim h).

const ROLES = [['h', 'Horizontal'], ['v', 'Vertical'], ['off', 'Off']];
const defaultRoles = (dims) => Object.fromEntries(dims.map((d, i) => [d, i === 0 ? 'h' : 'v']));

// Choosing Horizontal moves the previous holder to Vertical (D-40).
function assignRole(roles, dimNames, name, role) {
  if (role === 'h') for (const d of dimNames) if (d !== name && roles[d] === 'h') roles[d] = 'v';
  roles[name] = role;
  return roles;
}

// What a role means when a component is missing: no timeline -> 'h' is drawn as a slider; no sliders -> 'v' is off.
function effectiveRole(role, hasTimeline, hasSliders) {
  if (role === 'h' && !hasTimeline) role = 'v';
  if (role === 'v' && !hasSliders) role = 'off';
  return role;
}

// URL `slider.<dim>=h|v|off`, only for dims whose role differs from the default. Returns [[key, value], ...].
function roleParams(dimNames, roles) {
  const def = defaultRoles(dimNames);
  return dimNames.filter((d) => roles[d] && roles[d] !== def[d]).map((d) => [`slider.${d}`, roles[d]]);
}

// Parse the URL: roles by dim for the known dims with a valid value (unknown dims and values are ignored).
function readRoleParams(params, dimNames) {
  const out = {};
  for (const d of dimNames) {
    const role = params.get(`slider.${d}`);
    if (ROLES.some(([r]) => r === role)) out[d] = role;
  }
  return out;
}

registerComponent({
  name: 'dim-roles',
  init(app, root, opts) {
    const {state, bus} = app;
    const has = (n) => app.spec.components.some((c) => c.name === n);
    const hasTimeline = has('timeline'), hasSliders = has('dim-sliders');
    root.classList.add('gv-panel');
    root.hidden = true;

    // `slider.<dim>` URL entry; the defaults come from the union at write time. Two `h` keep the first (syncRoles).
    app.url.register('roles', {
      read(p) { if (!app.blank) Object.assign(state.roles, readRoleParams(p, Object.keys(app.cfg.dims))); },
      write(q) {
        if (app.blank) return;
        for (const [k, v] of roleParams(app.layers.unionDims(), state.roles)) q.set(k, v);
      },
    });

    function build() {
      root.replaceChildren();
      const dimNames = app.layers.unionDims();
      syncRoles(dimNames, state.roles);
      root.hidden = !dimNames.length;
      if (!dimNames.length) return;
      if (opts.title !== false) root.append(h('div', 'dr-title', 'Dimensions'));
      const roles = ROLES.filter(([r]) => (r === 'h' ? hasTimeline : r === 'v' ? hasSliders : true));
      for (const name of dimNames) {
        const sel = h('select');
        for (const [role, text] of roles) sel.add(new Option(text, role));
        sel.value = effectiveRole(state.roles[name], hasTimeline, hasSliders);
        sel.setAttribute('aria-label', `Role of ${name}`);
        sel.dataset.dim = name;
        sel.addEventListener('change', () => {
          assignRole(state.roles, dimNames, name, sel.value);
          app.url.sync();
          bus.emit('roles:changed', state.roles);  // timeline / dim-sliders (and this list) rebuild; playback stops
          root.querySelector(`select[data-dim="${CSS.escape(name)}"]`)?.focus();  // keep keyboard focus (D-40)
        });
        const row = h('label', 'dr-row');
        row.append(h('span', 'dr-name', name), sel);
        root.append(row);
      }
    }
    bus.on('config:loaded', build);
    bus.on('selection:changed', build);
    bus.on('roles:changed', build);
  },
});
}

// ---- component: center-lon
{
// Centre longitude (D-26): the world is drawn around this longitude (shifted layer copies). Reference: initCenterLon.
// Global datasets only; app.layers.setCenterLon normalises, stores and reloads the page (no in-place view rebuild).

// "190" -> -170; null for text that is not a number (the input then shows the current value again)
function parseLon(text) {
  const x = parseFloat(text);
  return Number.isFinite(x) ? normLon(x) : null;
}

registerComponent({
  name: 'center-lon',
  init(app, root) {
    root.classList.add('gv-panel', 'gv-row');
    root.hidden = true;
    const inp = h('input');
    inp.type = 'number'; inp.step = 'any'; inp.id = 'gv-center-lon';
    inp.setAttribute('aria-label', 'Centre longitude');
    const lab = h('label', '', 'Centre longitude');
    lab.htmlFor = inp.id;
    root.append(lab, inp);
    // the centre is read from the URL / storage when the layers start, i.e. after config:loaded
    const show = () => { root.hidden = !app.cfg.global; inp.value = String(app.state.centerLon); };
    app.bus.on('config:loaded', show);
    app.bus.on('map:ready', show);
    inp.addEventListener('change', () => {
      const lon = parseLon(inp.value);
      inp.value = String(lon ?? app.state.centerLon);
      if (lon !== null) app.layers.setCenterLon(lon);
    });
  },
});
}

// ---- component: basemap
{
// Basemap selector. Reference: initBasemap (the selector; the basemap layers and setBasemap live in core/layers.js).
registerComponent({
  name: 'basemap',
  init(app, root, opts) {
    root.classList.add('gv-panel', 'gv-row');
    const sel = h('select');
    sel.id = 'gv-basemap';
    sel.setAttribute('aria-label', 'Basemap');
    for (const [id, label] of app.layers.basemaps) sel.add(new Option(label, id));
    if (opts.label !== false) {
      const lab = h('label', '', 'Basemap');
      lab.htmlFor = sel.id;
      root.append(lab);
    }
    root.append(sel);
    sel.addEventListener('change', () => app.layers.setBasemap(sel.value));
    app.bus.on('basemap:changed', (id) => { sel.value = id; });
    app.layers.setBasemap(opts.default);
  },
});
}

// ---- component: cmap-sources
{
// Colourmap sources: which libraries the colourmap selects list (D-42), at least one on (D-47).
// Reference: initCmapToggle, syncCmapBoxes, setCmapShow. The state and the rule live in app.style.setSourceShown.

const CMAP_SOURCES = [['core', 'matplotlib'], ['cmocean', 'cmocean'], ['cmasher', 'cmasher']];
const ONLY_ONE = 'At least one colormap source must stay selected';

// syncCmapBoxes: checkbox state per source for `show` ({core, cmocean, cmasher} -> bool); the only source still on
// is disabled so it cannot be switched off.
function boxStates(show) {
  const only = CMAP_SOURCES.filter(([src]) => show[src]).length === 1;
  return Object.fromEntries(CMAP_SOURCES.map(([src]) => {
    const disabled = only && !!show[src];
    return [src, {checked: !!show[src], disabled, title: disabled ? ONLY_ONE : ''}];
  }));
}

registerComponent({
  name: 'cmap-sources',
  init(app, root) {
    root.classList.add('gv-panel');
    root.hidden = true;  // blank viewer: no colourmap selects
    root.append(h('div', 'cs-title', 'Colormaps'));
    const row = h('div', 'cs-boxes');
    root.append(row);
    const boxes = {};
    for (const [src, text] of CMAP_SOURCES) {
      const lab = h('label', 'cs-check'), box = h('input');
      box.type = 'checkbox';
      lab.append(box, h('span', '', text));
      row.append(lab);
      boxes[src] = box;
      // the core refuses to switch the last source off; sync() then puts the box back
      box.addEventListener('change', () => { app.style.setSourceShown(src, box.checked); sync(); });
    }
    const sync = () => {
      const states = boxStates(app.style.sources);
      for (const [src, box] of Object.entries(boxes)) Object.assign(box, states[src]);
    };
    app.bus.on('cmaps:changed', sync);
    app.bus.on('config:loaded', (cfg) => { root.hidden = !cfg.variables.length; });
    sync();
  },
});
}

// ---- component: style-controls
{
// Style controls of one variable: colourmap, Reversed, Log scale, Range, Auto | Min/Max | Symmetric, opacity.
// Reference: buildStyleControls (D-42 grouped colourmaps, D-45 Space reverses, D-48..D-51 range buttons, D-54 log).
// State lives in app.style / app.layers / app.frame; this file is only the DOM and the pure helpers below.

const AUTO_TITLE = 'Percentile range (2-98 %) of the data in view; a diverging colormap when the signs balance';
const MINMAX_TITLE = 'True minimum and maximum of the data in view';
const AUTO_NONE = 'No loaded data for the current frame';
const CMAP_TITLE = 'Tip: with this menu focused, press ← / → to step to the previous / next colormap, or Space to toggle Reversed.';

// stepSelect: the option value one step left/right of `current` in `values` (no wrap); null = no step (other key,
// modifier, already at the end, unknown value).
function stepValue(values, current, e) {
  if (hasModifier(e) || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return null;
  const i = values.indexOf(current);
  if (i < 0) return null;
  const j = Math.min(values.length - 1, Math.max(0, i + (e.key === 'ArrowRight' ? 1 : -1)));
  return j === i ? null : values[j];
}

// reverseKey (D-45): is this the plain Space key?
const isReverseKey = (e) => !hasModifier(e) && (e.key === ' ' || e.code === 'Space');

// setRange: both fields must be finite numbers with min < max, else null (the controls flash and keep the old values).
function parseRange(minText, maxText) {
  const a = parseFloat(minText), b = parseFloat(maxText);
  return Number.isFinite(a) && Number.isFinite(b) && a < b ? {vmin: a, vmax: b} : null;
}

// Symmetric button: [-m, m] with m = the larger magnitude; null when both limits are 0.
function symmetricRange(st) {
  const m = Math.max(Math.abs(st.vmin), Math.abs(st.vmax));
  return m > 0 ? {vmin: -m, vmax: m} : null;
}

// sync(): what the controls show for a style. `effLog` = the effective log flag (D-54: needs Min > 0).
function controlState(st, effLog) {
  return {
    reversed: !!st.reversed, log: !!st.log,
    logDisabled: !(st.vmin > 0),         // the flag is kept, it just has no effect
    symDisabled: !!effLog,               // symmetric about 0 is meaningless on a log scale
    minText: String(+st.vmin.toPrecision(6)), maxText: String(+st.vmax.toPrecision(6)),
  };
}

// Build the controls of one variable into a div. `getName()` says which variable they drive (fixed in a card, the
// panel's select otherwise). Returns {el, sync, dispose}.
function buildControls(app, opts, getName) {
  const el = h('div', 'gv-style-controls sc-controls');
  const row = (label) => {
    const r = h('div', 'sc-row');
    if (label) r.append(h('label', '', label));
    el.append(r);
    return r;
  };
  const S = app.style;

  // colourmap select (grouped by source, D-42) + gradient preview
  const sel = h('select');
  sel.setAttribute('aria-label', 'Colormap');
  sel.title = CMAP_TITLE;
  const bar = h('span', 'sc-cmbar');
  row('Colormap').append(sel, bar);
  const checks = row('');
  checks.classList.add('sc-checks');
  const check = (text, title) => {
    const lab = h('label', 'sc-check'), box = h('input');
    box.type = 'checkbox';
    if (title) lab.title = title;
    lab.append(box, h('span', '', text));
    checks.append(lab);
    return box;
  };
  const rev = check('Reversed');
  const logBox = opts.log !== false ? check('Log scale', 'Space the colours logarithmically (needs Min > 0)') : null;

  const minIn = h('input'), maxIn = h('input');
  for (const [inp, name] of [[minIn, 'Min'], [maxIn, 'Max']]) { inp.type = 'number'; inp.step = 'any'; inp.setAttribute('aria-label', name); }
  row('Range').append(minIn, h('span', 'sc-sep', ':'), maxIn);

  let autoBtn, minMaxBtn, symBtn;
  if (opts.range_buttons !== false) {
    const r = row('');
    autoBtn = h('button', 'gv-btn', 'Auto'); minMaxBtn = h('button', 'gv-btn', 'Min/Max'); symBtn = h('button', 'gv-btn', 'Symmetric');
    for (const b of [autoBtn, minMaxBtn, symBtn]) b.type = 'button';
    autoBtn.title = AUTO_TITLE; minMaxBtn.title = MINMAX_TITLE;
    r.append(autoBtn, minMaxBtn, symBtn);
  }
  let opac;
  if (opts.opacity !== false) {
    opac = h('input');
    opac.type = 'range'; opac.min = '0'; opac.max = '1'; opac.step = '0.05';
    opac.setAttribute('aria-label', 'Opacity');
    row('Opacity').append(opac);
  }

  // fillCmapSelect: groups of the shown sources; the current map stays listed even from a hidden source
  const fill = (current) => {
    const groups = S.listedGroups(current).map((g) => {
      const og = document.createElement('optgroup');
      og.label = g.label;
      for (const n of g.names) og.append(new Option(S.cmapLabel(n), n));
      return og;
    });
    sel.replaceChildren(...groups);
    sel.value = current;
  };
  const sync = () => {
    const name = getName(), st = name && S.get(name);
    if (!st) return;
    const c = controlState(st, S.effectiveLog(st));
    fill(st.cmap);
    rev.checked = c.reversed;
    if (logBox) { logBox.checked = c.log; logBox.disabled = c.logDisabled; }
    if (symBtn) symBtn.disabled = c.symDisabled;
    minIn.value = c.minText; maxIn.value = c.maxText;
    bar.style.background = S.gradientCss(S.stopsOf(st));
    if (opac) opac.value = String(app.layers.opacityOf(name));
  };
  const flashBad = () => {
    for (const i of [minIn, maxIn]) i.classList.add('bad');
    setTimeout(() => { minIn.classList.remove('bad'); maxIn.classList.remove('bad'); }, 1500);
  };
  // a style call that may be rejected (invalid range): on failure show the old values again and flash
  const apply = (patch) => { if (!S.set(getName(), patch)) { sync(); flashBad(); } };

  sel.addEventListener('change', () => apply({cmap: sel.value, reversed: false}));
  // ←/→ step across optgroups (native stepping is unreliable on macOS Chrome); Space toggles Reversed (D-45).
  sel.addEventListener('keydown', (e) => {
    const next = stepValue(Array.from(sel.options, (o) => o.value), sel.value, e);
    if (next !== null) {
      e.preventDefault(); e.stopPropagation();
      sel.value = next;
      sel.dispatchEvent(new Event('change', {bubbles: true}));
    }
  });
  // Space: swallow keydown, keyup and auto-repeat (some browsers open the list on keyup); toggle once per press
  const spaceKey = (e) => {
    if (!isReverseKey(e)) return;
    e.preventDefault(); e.stopPropagation();
    if (e.type === 'keydown' && !e.repeat) rev.click();
  };
  sel.addEventListener('keydown', spaceKey);
  sel.addEventListener('keyup', spaceKey);
  rev.addEventListener('change', () => apply({reversed: rev.checked}));
  if (logBox) logBox.addEventListener('change', () => apply({log: logBox.checked}));
  const setRange = () => {
    const r = parseRange(minIn.value, maxIn.value);
    if (r) apply(r); else { sync(); flashBad(); }
  };
  minIn.addEventListener('change', setRange);
  maxIn.addEventListener('change', setRange);
  if (autoBtn) {
    // nothing loaded for the current frame: the style is kept (D-48, D-51)
    const run = (btn, title, fn) => btn.addEventListener('click', () => {
      if (fn(getName())) { btn.title = title; return; }
      btn.title = AUTO_NONE;
      flashBad();
    });
    run(autoBtn, AUTO_TITLE, (n) => app.frame.autoStyle(n));
    run(minMaxBtn, MINMAX_TITLE, (n) => app.frame.minMaxStyle(n));
    symBtn.addEventListener('click', () => { const r = symmetricRange(S.get(getName())); if (r) apply(r); });
  }
  if (opac) opac.addEventListener('input', () => app.layers.setOpacity(getName(), +opac.value));

  const refresh = (name) => { if (name === undefined || name === getName()) sync(); };
  const unsub = [app.bus.on('style:changed', refresh), app.bus.on('opacity:changed', refresh),
    app.bus.on('cmaps:changed', () => refresh())];
  sync();
  return {el, sync, dispose: () => unsub.forEach((f) => f())};
}

registerComponent({
  name: 'style-controls',
  init(app, root, opts) {
    if (opts.placement === 'panel') {
      // own panel: one variable at a time, picked in a small select (follows the selection; labels per label mode)
      root.classList.add('gv-panel', 'sc-panel');
      root.hidden = true;
      const sel = h('select');
      sel.setAttribute('aria-label', 'Variable');
      const head = h('div', 'sc-row');
      head.append(h('label', '', 'Variable'), sel);
      const note = h('div', 'sc-note', 'No style controls for categorical variables.');
      note.hidden = true;
      let current = null;
      const ctl = buildControls(app, opts, () => current);
      root.append(head, ctl.el, note);
      const fillVars = () => {
        const vars = app.cfg.variables;
        root.hidden = !vars.length;
        sel.replaceChildren(...vars.map((v) => new Option(app.label(v), v.name)));
        if (!current || !app.varOf(current)) current = app.state.selected.at(-1) || vars[0]?.name || null;
        if (current) sel.value = current;
        show();
      };
      const show = () => {
        const v = current && app.varOf(current);
        ctl.el.hidden = !v || !!v.categorical;
        note.hidden = !(v && v.categorical);
        if (v && !v.categorical) ctl.sync();
      };
      sel.addEventListener('change', () => { current = sel.value; show(); });
      app.bus.on('config:loaded', fillVars);
      app.bus.on('labels:changed', fillVars);
      // the newly selected (top-most) variable becomes the one shown
      app.bus.on('selection:changed', (selected) => { if (selected.length) { current = selected.at(-1); sel.value = current; show(); } });
    } else {
      // card: fill the settings container of each variable card (re-emitted containers replace the old controls)
      root.hidden = true;
      const mine = new globalThis.Map();  // name -> controls
      app.bus.on('card:settings', ({name, el}) => {
        mine.get(name)?.el.remove();
        mine.get(name)?.dispose();
        const v = app.varOf(name);
        if (!v || v.categorical) return;
        const ctl = buildControls(app, opts, () => name);
        el.append(ctl.el);
        mine.set(name, ctl);
      });
    }
  },
});
}

// ---- component: legend
{
// Colourbar legend: gradient, ticks (log ticks under an effective log scale, D-54) and units.
// Reference: buildLegend, layoutTicks (D-54, D-44: no variable name in the card legend; overlays add a title).

// Ticks of a style as [{pos, label}], pos = fraction 0..1 along the bar. `style` supplies effectiveLog, niceTicks and
// niceLogTicks (app.style). buildLegend: linear (t - vmin)/(vmax - vmin), log ln(t/vmin)/ln(vmax/vmin).
function legendTicks(st, style) {
  const log = style.effectiveLog(st);
  const values = log ? style.niceLogTicks(st.vmin, st.vmax) : style.niceTicks(st.vmin, st.vmax);
  return values.map((t) => {
    const f = log ? Math.log(t / st.vmin) / Math.log(st.vmax / st.vmin) : (t - st.vmin) / (st.vmax - st.vmin);
    return {pos: clamp(f, 0, 1), label: fmt(t)};
  });
}

// layoutTicks: labels are centred on their tick; the end ones are nudged back inside the bar, a label that would
// touch the previous one is hidden. `items` = [{start, size}] in screen order along the axis (start = left/top of the
// centred label), `lo`/`hi` = the bar's extent. Returns [{shift, hidden}].
function fitLabels(items, lo, hi, gap = 4) {
  let end = -Infinity;
  return items.map(({start, size}) => {
    const shift = start < lo ? lo - start : (start + size > hi ? hi - (start + size) : 0);
    const s = start + shift;
    if (s < end + gap) return {shift, hidden: true};
    end = s + size;
    return {shift, hidden: false};
  });
}

// the colourbar gradient runs left to right in app.style.gradientCss; vertical bars put the high end on top
const gradientFor = (css, vertical) => (vertical ? css.replace('to right', 'to top') : css);

// DOM wrapper around fitLabels for one .lg-ticks element
function layoutTicks(ticks, vertical) {
  const box = ticks.getBoundingClientRect();
  const len = vertical ? box.height : box.width;
  if (!len) return;  // sidebar collapsed / settings closed / not laid out yet
  const labs = Array.from(ticks.querySelectorAll('.lg-label'));
  const dir = vertical ? 'Y' : 'X';
  const origin = vertical ? box.top : box.left;
  for (const lab of labs) { lab.style.visibility = 'visible'; lab.style.transform = `translate${dir}(-50%)`; }
  // screen order along the axis (a vertical bar has the high ticks on top)
  const items = labs.map((lab) => {
    const r = lab.getBoundingClientRect();
    return {lab, start: (vertical ? r.top : r.left) - origin, size: vertical ? r.height : r.width};
  }).sort((a, b) => a.start - b.start);
  fitLabels(items, 0, len).forEach((f, i) => {
    const lab = items[i].lab;
    if (f.shift) lab.style.transform = `translate${dir}(calc(-50% + ${f.shift}px))`;
    lab.style.visibility = f.hidden ? 'hidden' : 'visible';
  });
}

// One legend element for a variable. title: caption above (overlay only).
function buildLegend(app, v, opts, vertical, title) {
  const el = h('div', `lg-legend ${vertical ? 'lg-vertical' : 'lg-horizontal'}`);
  if (title) el.append(h('div', 'lg-title', title));
  if (v.categorical) {  // palette swatches
    for (const [val, c] of v.palette) {
      const sw = h('span', 'lg-swatch');
      sw.style.background = css(c);
      const r = h('div', 'lg-cat');
      r.append(sw, h('span', '', String(val)));
      el.append(r);
    }
    return el;
  }
  const st = app.style.get(v.name);
  const body = h('div', 'lg-body');
  const bar = h('div', 'lg-bar');
  bar.style.background = gradientFor(app.style.gradientCss(app.style.stopsOf(st)), vertical);
  const ticks = h('div', 'lg-ticks');
  for (const {pos, label} of legendTicks(st, app.style)) {
    const p = `${100 * (vertical ? 1 - pos : pos)}%`;
    const mark = h('span', 'lg-tick'), lab = h('span', 'lg-label', label);
    mark.style[vertical ? 'top' : 'left'] = lab.style[vertical ? 'top' : 'left'] = p;
    ticks.append(mark, lab);
  }
  body.append(bar, ticks);
  el.append(body);
  const units = unitsOf(v);
  if (opts.units !== false && units) el.append(h('div', 'lg-units', units));
  new ResizeObserver(() => layoutTicks(ticks, vertical)).observe(ticks);
  return el;
}

registerComponent({
  name: 'legend',
  init(app, root, opts) {
    root.hidden = true;
    if (opts.placement === 'card') {
      // inside each card's settings container; only that variable's style changes redraw it
      const mine = new globalThis.Map();  // name -> {legend, draw}
      app.bus.on('card:settings', ({name, el}) => {
        mine.get(name)?.legend.remove();
        const v = app.varOf(name);
        if (!v) return;
        const wrap = h('div', 'gv-legend lg-card');
        const draw = () => wrap.replaceChildren(buildLegend(app, v, opts, false));
        draw();
        el.append(wrap);
        mine.set(name, {legend: wrap, draw});
      });
      app.bus.on('style:changed', (name) => mine.get(name)?.draw());
      return;
    }
    // overlay: one colourbar per visible variable, top-most first, titled with its label
    const vertical = opts.orientation === 'vertical';
    root.classList.add('lg-overlay', vertical ? 'lg-vertical' : 'lg-horizontal');
    const draw = () => {
      const vars = [...app.state.selected].reverse().filter((n) => !app.state.hidden.has(n)).map((n) => app.varOf(n)).filter(Boolean);
      root.replaceChildren(...vars.map((v) => {
        const panel = h('div', 'gv-panel lg-panel');
        panel.append(buildLegend(app, v, opts, vertical, app.label(v)));
        return panel;
      }));
      root.hidden = !vars.length;
    };
    for (const ev of ['selection:changed', 'visibility:changed', 'style:changed', 'labels:changed', 'config:loaded']) app.bus.on(ev, draw);
  },
});
}

// ---- component: var-attrs
{
// Per-variable attributes popover (D-55). Reference: toggleVarInfo, closeVarInfo, initVarInfo. The info icon belongs to
// variable-cards, which emits varinfo:open({name, anchor}); this component owns one shared popover (position: fixed,
// so its slot does not matter; the root itself stays empty). Attribute values arrive formatted in cfg (attrValue).

// Rows of the table: [key, text] in stored order.
function attrRows(attrs) {
  return Object.entries(attrs || {}).map(([k, x]) => [k, String(x)]);
}

// Reference toggleVarInfo placement: right of the sidebar (or of the anchor), vertically at the anchor, kept inside
// the viewport (and below `minTop`, e.g. a top bar). Rects are {top, right}; sizes in px.
function popoverPosition(at, side, panel, view, minTop = 8) {
  const left = Math.max(8, Math.min(Math.max(side ? side.right : 0, at.right) + 8, view.w - panel.w - 8));
  const top = Math.max(minTop, Math.min(at.top, view.h - panel.h - 8));
  return {left, top};
}

registerComponent({
  name: 'var-attrs',
  init(app, root) {
    const panel = h('div', 'gv-panel va-pop');
    panel.hidden = true;
    root.append(panel);
    let current = null, anchorEl = null;

    function close() {
      panel.hidden = true;
      current = anchorEl = null;
      app.bus.emit('varinfo:state', {name: null});
    }
    function open(name, anchor) {
      const v = app.varOf(name);
      if (!v) return;
      panel.replaceChildren(h('div', 'va-title', app.label(v)));
      const rows = attrRows(v.attrs);
      if (rows.length) {
        panel.append(kvTable(rows));
      } else {
        panel.append(h('div', 'va-none', 'No attributes'));
      }
      panel.hidden = false;
      const at = anchor.getBoundingClientRect();
      const side = anchor.closest('.gv-sidebar')?.getBoundingClientRect();
      const top = app.slot('topbar')?.getBoundingClientRect();
      const pos = popoverPosition(at, side, {w: panel.offsetWidth, h: panel.offsetHeight},
        {w: innerWidth, h: innerHeight}, Math.max(8, top ? top.bottom + 4 : 0));
      panel.style.left = `${pos.left}px`;
      panel.style.top = `${pos.top}px`;
      current = name;
      anchorEl = anchor;
      app.bus.emit('varinfo:state', {name});
    }

    // clicking the same card's icon again closes it
    app.bus.on('varinfo:open', ({name, anchor}) => { const same = current === name; close(); if (!same) open(name, anchor); });
    // capture phase: card buttons stop propagation, so a bubbling listener would miss them
    document.addEventListener('click', (e) => {
      if (current && !panel.contains(e.target) && !anchorEl.contains(e.target)) close();
    }, true);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && current) close(); });
    app.bus.on('selection:changed', () => { if (current && !app.state.selected.includes(current) && !anchorEl.isConnected) close(); });
    app.bus.on('labels:changed', () => { if (current) open(current, anchorEl); });
  },
});
}

// ---- component: timeline
{
// Horizontal timeline for the dim whose role is 'h'. Reference: buildScrubber/buildTimeline (rebuilt when the dims
// or roles change), stepPositions, dateTicks, indexTicks, stopPlay, scrubKeyDown (D-50), syncRoles (D-40, core/util.js).
// Roles live in app.state.roles ('h' | 'v' | 'off'); without dim-roles the defaults apply (first dim h).

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// Position of every step along the rail, 0..1: proportional to time for ISO dims, else evenly spaced.
function stepPositions(d) {
  const n = d.size;
  if (d.iso && d.iso.length === n && n > 1) {
    const t = d.iso.map((s) => Date.parse(s));
    if (t.every(Number.isFinite)) {
      const lo = Math.min(...t), hi = Math.max(...t);
      if (hi > lo) return {pos: t.map((x) => (x - lo) / (hi - lo)), timed: true, lo, hi};
    }
  }
  return {pos: Array.from({length: n}, (_, i) => (n > 1 ? i / (n - 1) : 0)), timed: false};
}

// Major ticks of a time axis: years (> 2 years of data) or months, about 10 labels at most.
function dateTicks(lo, hi) {
  const out = [];
  const spanDays = (hi - lo) / 864e5;
  const first = new Date(lo);
  if (spanDays > 730) {
    const years = spanDays / 365.25;
    const step = [1, 2, 5, 10, 20, 50, 100].find((s) => years / s <= 10) || 100;
    let y = first.getUTCFullYear();
    if (Date.UTC(y, 0, 1) < lo) y += 1;
    y = Math.ceil(y / step) * step;
    for (; Date.UTC(y, 0, 1) <= hi; y += step) out.push({t: Date.UTC(y, 0, 1), text: String(y)});
  } else {
    const step = [1, 2, 3, 6].find((s) => spanDays / 30.4 / s <= 12) || 6;
    const y = first.getUTCFullYear();
    let m = first.getUTCMonth();
    if (Date.UTC(y, m, 1) < lo) m += 1;
    for (; Date.UTC(y, m, 1) <= hi; m += 1) {
      if ((y * 12 + m) % step) continue;
      const mm = m % 12;
      out.push({t: Date.UTC(y, m, 1), text: MONTHS[mm] + (mm === 0 || !out.length ? ` ${y + Math.floor(m / 12)}` : '')});
    }
  }
  return out;
}

// Major ticks for a plain index (or non-time label) axis: every 1/2/5 x 10^k steps, about 8 labels.
function indexTicks(d, pos) {
  const n = d.size;
  const raw = Math.max(1, n / 8);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let i = 0; i < n; i += step) out.push({p: pos[i], text: dimLabel(d, i)});
  return out;
}

// Index of the step nearest to x (0..1 along the rail), as in the reference's `nearest`.
function nearestStep(pos, x) {
  let best = 0;
  for (let i = 1; i < pos.length; i++) if (Math.abs(pos[i] - x) < Math.abs(pos[best] - x)) best = i;
  return best;
}

registerComponent({
  name: 'timeline',
  init(app, root, opts) {
    const {state, bus} = app;
    root.classList.add('gv-panel');
    root.hidden = true;
    let timer = null, cur = null;  // cur = {name, step} of the built timeline

    const stopPlay = () => {
      if (timer) clearInterval(timer);
      timer = null;
      const b = root.querySelector('.tl-play');
      if (b) { b.innerHTML = ICON.play; b.setAttribute('aria-label', 'Play'); }
    };

    function build() {
      stopPlay();
      cur = null;
      root.replaceChildren();
      const dimNames = app.layers.unionDims();
      syncRoles(dimNames, state.roles);
      const name = dimNames.find((d) => state.roles[d] === 'h');
      root.hidden = !name;
      if (!name) return;
      const d = app.cfg.dims[name], n = d.size;
      state.dims[name] = clamp(state.dims[name] || 0, 0, n - 1);
      const {pos, timed, lo, hi} = stepPositions(d);
      const mkBtn = (icon, label, cls) => {
        const b = h('button', `gv-iconbtn ${cls || ''}`); b.type = 'button'; b.innerHTML = icon; b.setAttribute('aria-label', label); return b;
      };
      const ctl = h('div', 'tl-ctl');
      const prevBtn = mkBtn(ICON.prev, 'Previous step'), nextBtn = mkBtn(ICON.next, 'Next step');
      const playBtn = opts.play !== false ? mkBtn(ICON.play, 'Play', 'tl-play') : null;
      ctl.append(prevBtn, ...(playBtn ? [playBtn] : []), nextBtn);

      const track = h('div', 'tl-track');
      track.append(h('div', 'tl-rail'));
      const addMark = (p, cls) => { const m = h('div', `tl-mark ${cls}`); m.style.left = `${100 * p}%`; track.append(m); };
      if (n <= 60) pos.forEach((p) => addMark(p, 'minor'));
      let major = [];
      if (timed) major = dateTicks(lo, hi).map((k) => ({p: (k.t - lo) / (hi - lo), text: k.text}));
      if (major.length < 2) major = indexTicks(d, pos);
      const labels = [];
      for (const k of major) {
        addMark(k.p, 'major');
        const lab = h('div', 'tl-label', String(k.text));
        lab.style.left = `${100 * k.p}%`;
        track.append(lab);
        labels.push(lab);
      }
      const handle = h('div', 'tl-handle');
      handle.setAttribute('role', 'slider');
      handle.setAttribute('aria-label', name);
      handle.setAttribute('aria-valuemin', '0');
      handle.setAttribute('aria-valuemax', String(n - 1));
      const bubble = h('div', 'tl-bubble');
      track.append(handle, bubble);
      root.append(ctl, h('span', 'tl-name', name), track);

      const render = () => {
        const i = state.dims[name];
        handle.style.left = bubble.style.left = `${100 * pos[i]}%`;
        bubble.textContent = String(dimLabel(d, i));
        handle.setAttribute('aria-valuenow', String(i));
        handle.setAttribute('aria-valuetext', String(dimLabel(d, i)));
      };
      const setStep = (i) => {
        i = clamp(i, 0, n - 1);
        if (i !== state.dims[name]) app.layers.setDim(name, i);  // emits dims:changed -> render
        render();
      };
      const step = (delta) => setStep(state.dims[name] + delta);
      let dragging = false;
      const nearest = (clientX) => {
        const r = track.getBoundingClientRect();
        return nearestStep(pos, r.width ? clamp((clientX - r.left) / r.width, 0, 1) : 0);
      };
      track.addEventListener('pointerdown', (e) => {
        dragging = true;
        track.setPointerCapture?.(e.pointerId);
        handle.style.cursor = 'grabbing';
        stopPlay();
        setStep(nearest(e.clientX));
      });
      track.addEventListener('pointermove', (e) => { if (dragging) setStep(nearest(e.clientX)); });
      const end = () => { dragging = false; handle.style.cursor = ''; };
      track.addEventListener('pointerup', end);
      track.addEventListener('pointercancel', end);
      prevBtn.addEventListener('click', () => { stopPlay(); step(-1); });
      nextBtn.addEventListener('click', () => { stopPlay(); step(1); });
      playBtn?.addEventListener('click', () => {
        if (timer) return stopPlay();
        playBtn.innerHTML = ICON.pause;
        playBtn.setAttribute('aria-label', 'Pause');
        timer = setInterval(() => setStep((state.dims[name] + 1) % n), 1000 / opts.fps);
      });

      // hide tick labels that would collide with the previous one
      const layoutLabels = () => {
        let right = -Infinity;
        for (const lab of labels) {
          lab.style.visibility = 'visible';
          const r = lab.getBoundingClientRect();
          if (r.left < right + 6) lab.style.visibility = 'hidden';
          else right = r.right;
        }
      };
      new ResizeObserver(layoutLabels).observe(track);
      cur = {name, step, render};
      render();
      layoutLabels();
    }

    // Arrow keys (D-50): Left/Right step the horizontal dim; ignored while typing in a field or with modifiers.
    if (opts.keys !== false) {
      document.addEventListener('keydown', (e) => {
        if (!cur || !plainKey(e)) return;
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        e.preventDefault();
        stopPlay();
        cur.step(e.key === 'ArrowRight' ? 1 : -1);
      });
      // dim-sliders asks for the same step when Left/Right is pressed on a focused vertical slider
      bus.on('timeline:step', (delta) => { if (cur) { stopPlay(); cur.step(delta); } });
    }

    bus.on('config:loaded', build);
    bus.on('selection:changed', build);
    bus.on('roles:changed', build);
    bus.on('dims:changed', ({dim}) => { if (cur && cur.name === dim) cur.render(); });
  },
});
}

// ---- component: dim-sliders
{
// Sliders for the dims whose role is 'v'. Reference: buildDimSlider, dimKeyDown, scrubKeyDown (D-50), syncRoles (D-40).
// Roles live in app.state.roles; without dim-roles the defaults apply (first dim h, the rest v). Without a timeline
// component nothing renders the 'h' dim, so it is treated as vertical here.


// The dims this component draws, in union order: role 'v', plus 'h' when no timeline exists.
function verticalDims(dimNames, roles, hasTimeline) {
  return dimNames.filter((d) => roles[d] === 'v' || (roles[d] === 'h' && !hasTimeline));
}

registerComponent({
  name: 'dim-sliders',
  init(app, root, opts) {
    const {state, bus} = app;
    const hasTimeline = app.spec.components.some((c) => c.name === 'timeline');
    const horizontal = opts.orientation === 'horizontal';
    root.dataset.orientation = horizontal ? 'horizontal' : 'vertical';
    root.hidden = true;
    let steps = [];  // [{name, step(delta)}] in display order

    function build() {
      steps = [];
      root.replaceChildren();
      const dimNames = app.layers.unionDims();
      syncRoles(dimNames, state.roles);
      const vDims = verticalDims(dimNames, state.roles, hasTimeline);
      root.hidden = !vDims.length;
      for (const name of vDims) {
        const d = app.cfg.dims[name];
        state.dims[name] = clamp(state.dims[name] || 0, 0, d.size - 1);
        const item = h('div', 'gv-panel ds-item');
        const val = h('span', 'ds-val', String(dimLabel(d, state.dims[name])));
        const inp = h('input');
        inp.type = 'range'; inp.min = 0; inp.max = d.size - 1; inp.step = 1; inp.value = state.dims[name];
        if (!horizontal) inp.setAttribute('orient', 'vertical');  // minimum at the top (writing-mode in the CSS)
        inp.setAttribute('aria-label', name);
        const apply = () => { app.layers.setDim(name, +inp.value); val.textContent = String(dimLabel(d, +inp.value)); };
        inp.addEventListener('input', apply);
        // Keyboard stepping (D-50): the same path as a native input event, clamped at both ends.
        steps.push({name, inp, step: (delta) => {
          const i = clamp(+inp.value + delta, 0, d.size - 1);
          if (i === +inp.value) return;
          inp.value = i;
          apply();
        }});
        // Focused slider: Up/Down step it, Left/Right the timeline (via the bus); Home/End/Page keys stay native.
        inp.addEventListener('keydown', (e) => {
          if (opts.keys === false || hasModifier(e)) return;
          if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
            e.preventDefault(); e.stopPropagation();
            steps.find((s) => s.name === name).step(e.key === 'ArrowDown' ? 1 : -1);  // Up = index - 1
          } else if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && hasTimeline) {
            e.preventDefault(); e.stopPropagation();
            bus.emit('timeline:step', e.key === 'ArrowRight' ? 1 : -1);
          }
        });
        item.append(h('span', 'ds-lab', name), inp, val);
        root.append(item);
      }
    }

    // global Up/Down: the first slider (D-50); fields and modifiers are left alone
    if (opts.keys !== false) {
      document.addEventListener('keydown', (e) => {
        if (!steps.length || !plainKey(e)) return;
        if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
        e.preventDefault();
        steps[0].step(e.key === 'ArrowDown' ? 1 : -1);
      });
    }

    bus.on('config:loaded', build);
    bus.on('selection:changed', build);
    bus.on('roles:changed', build);
    bus.on('dims:changed', ({dim, index}) => {  // changed from elsewhere (URL, timeline of another dim is irrelevant)
      const s = steps.find((x) => x.name === dim);
      if (!s || +s.inp.value === index) return;
      s.inp.value = index;
      s.inp.nextSibling.textContent = String(dimLabel(app.cfg.dims[dim], index));
    });
  },
});
}

// ---- component: readout
{
// Pointer readout: value of every visible variable under the pointer plus lat/lon. Reference: initReadout, coordLine
// (D-39/D-41). It sits in its slot instead of following the pointer; the sampling itself is app.frame.sample.

// "Lat: 46.520°N  Lon: 6.630°E" (lon normalised to [-180, 180) by the caller). No "0.000°S".
function formatLatLon(lon, lat) {
  const f = (v, p, n) => `${Math.abs(v).toFixed(3)}°${+v.toFixed(3) < 0 ? n : p}`;
  return `Lat: ${f(lat, 'N', 'S')}  Lon: ${f(lon, 'E', 'W')}`;
}

// "name: 12.34 unit" (categorical values print as they are, without units)
function formatValue(val, v, label) {
  const u = unitsOf(v);
  return `${label}: ${v.categorical ? String(val) : +Number(val).toPrecision(4)}${!v.categorical && u ? ` ${u}` : ''}`;
}

registerComponent({
  name: 'readout',
  init(app, root, opts) {
    root.classList.add('gv-panel');
    root.hidden = true;
    const hide = () => { root.hidden = true; };
    app.bus.on('map:ready', (map) => {
      map.on('pointermove', (e) => {
        if (e.dragging) return hide();
        const samples = app.frame.sample(e.coordinate);  // top-most first
        const lines = samples.map((s) => formatValue(s.value, app.varOf(s.name), app.label(s.name)));
        let coord = null;
        // lat/lon inside the data extent, or anywhere while a value is shown (as in the reference's coordLine)
        if (opts.latlon !== false && (samples.length || app.geo.inExtent(e.coordinate))) {
          const ll = app.geo.lonLat(e.coordinate);
          if (ll) coord = formatLatLon(ll[0], ll[1]);
        }
        if (!lines.length && !coord) return hide();
        root.replaceChildren(...(coord ? [h('div', 'ro-coord', coord)] : []), ...lines.map((t) => h('div', 'ro-val', t)));
        root.hidden = false;
      });
      map.getViewport().addEventListener('pointerleave', hide);
    });
  },
});
}

bootViewer().catch((e) => { console.error(e); });
