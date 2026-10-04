import * as THREE from "three";

// World: x, y = the semantic map (similar messages sit together), z = time on a log scale around an anchor:
//   z = -scale * sign(anchor - t) * log1p(|anchor - t| / tau)
// Times much closer than tau to the anchor are plotted linearly, to the second; farther ones compress. The
// overview anchors at the newest message, so the last weeks spread out up close and the first months become
// the far haze. Focusing a turn anchors at its moment with tau = 1 hour, so the minutes around it open into
// distance: what it read recedes ahead of you, what came after is behind you. z blends between the two.
const SPREAD = 60, DEPTH = 400, FOCUS_TAU = 1 / 24, FOCUS_SCALE = 180;
let tau = 10;                                   // overview tau, days: [ and ] change it

const [meta, starBuf, colorBuf, edgeBuf, echoIndexBuf, echoSimBuf, contextBuf, contextKindBuf] = await Promise.all([
  fetch("data/meta.json").then(r => r.json()),
  fetch("data/stars.bin").then(r => r.arrayBuffer()),
  fetch("data/colors.bin").then(r => r.arrayBuffer()),
  fetch("data/edges.bin").then(r => r.arrayBuffer()),
  fetch("data/echo_index.bin").then(r => r.ok ? r.arrayBuffer() : null),
  fetch("data/echo_sim.bin").then(r => r.ok ? r.arrayBuffer() : null),
  fetch("data/context.bin").then(r => r.ok ? r.arrayBuffer() : null),
  fetch("data/context_kinds.bin").then(r => r.ok ? r.arrayBuffer() : null),
]);
const t0 = Date.parse(meta.first.replace(" ", "T") + "Z");
const n = meta.count, stars = new Float32Array(starBuf), rgbs = new Uint8Array(colorBuf);
const daysOf = i => stars[i * 4 + 2];
document.getElementById("counts").textContent =
  `${n.toLocaleString()} turns · ${meta.speakers.length} speakers · ${meta.first.slice(0, 10)} to ${meta.last.slice(0, 10)}`;

const overviewLens = () => ({ anchor: meta.days, tau, scale: DEPTH / Math.log1p(meta.days / tau), squash: [1, 1], centre: [0, 0] });
// Focus re-anchors time at the turn, with tau fitted to its conversation: the median gap to what it read and
// who read it, so seconds-apart chatter spreads as evenly as hours-apart; the farthest lands 30 units out.
// It also squashes meaning toward the turn: seen from the side, semantic x is depth, and a message on another
// topic would otherwise sit between the camera and the turn.
function focusLens(i, around) {
  const gaps = around.map(j => Math.abs(daysOf(j) - daysOf(i))).filter(g => g > 0).sort((a, b) => a - b);
  const tau = Math.min(1, Math.max(10 / 86400, gaps.length ? gaps[gaps.length >> 1] : 1 / 24));
  const far = gaps.length ? Math.min(gaps[gaps.length - 1], 2) : 1 / 24;
  return { anchor: daysOf(i), tau, scale: 30 / Math.log1p(far / tau), squash: [0.2, 0.4],
    centre: [stars[i * 4] * SPREAD, stars[i * 4 + 1] * SPREAD] };
}
const xIn = (lens, i) => lens.centre[0] + (stars[i * 4] * SPREAD - lens.centre[0]) * lens.squash[0];
const yIn = (lens, i) => lens.centre[1] + (stars[i * 4 + 1] * SPREAD - lens.centre[1]) * lens.squash[1];
const depthIn = (lens, days) => {
  const d = lens.anchor - days;
  return -lens.scale * Math.sign(d) * Math.log1p(Math.abs(d) / lens.tau);
};
let lens = overviewLens();

const position = new Float32Array(n * 3), color = new Float32Array(n * 3), reads = new Float32Array(n);
for (let i = 0; i < n; i++) {
  position[i * 3] = stars[i * 4] * SPREAD;
  position[i * 3 + 1] = stars[i * 4 + 1] * SPREAD;
  position[i * 3 + 2] = depthIn(lens, daysOf(i));
  for (let c = 0; c < 3; c++) color[i * 3 + c] = rgbs[i * 4 + c] / 255;
  reads[i] = stars[i * 4 + 3];
}
// lens changes blend x and z from where each star is to where the new lens puts it
const fromX = new Float32Array(n), toX = new Float32Array(n), fromY = new Float32Array(n), toY = new Float32Array(n);
const fromZ = new Float32Array(n), toZ = new Float32Array(n);
let lensBlend = 1;
function setLens(next, instant) {
  lens = next;
  for (let i = 0; i < n; i++) {
    fromX[i] = position[i * 3]; toX[i] = xIn(lens, i);
    fromY[i] = position[i * 3 + 1]; toY[i] = yIn(lens, i);
    fromZ[i] = position[i * 3 + 2]; toZ[i] = depthIn(lens, daysOf(i));
  }
  lensBlend = instant ? 0.9999 : 0;
}
function stepLens(dt) {
  if (lensBlend >= 1) return false;
  lensBlend = Math.min(1, lensBlend + dt / 1.2);
  const e = lensBlend >= 1 ? 1 : lensBlend * lensBlend * (3 - 2 * lensBlend);
  for (let i = 0; i < n; i++) {
    position[i * 3] = fromX[i] + (toX[i] - fromX[i]) * e;
    position[i * 3 + 1] = fromY[i] + (toY[i] - fromY[i]) * e;
    position[i * 3 + 2] = fromZ[i] + (toZ[i] - fromZ[i]) * e;
  }
  positionAttr.needsUpdate = true;
  return true;
}

const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.05, 6000);

const positionAttr = new THREE.BufferAttribute(position, 3);
const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", positionAttr);
geometry.setAttribute("color", new THREE.BufferAttribute(color, 3));
geometry.setAttribute("reads", new THREE.BufferAttribute(reads, 1));

// the graph both ways, as CSR: out = edges to later turns, in = edges from earlier turns
const edges = new Uint32Array(edgeBuf), m = edges.length / 2;
function adjacency(pairs, from, to, kinds) {
  const count = pairs.length / 2, start = new Uint32Array(n + 1), list = new Uint32Array(count);
  const kind = kinds ? new Uint8Array(count) : null;
  for (let e = 0; e < count; e++) start[pairs[e * 2 + from] + 1]++;
  for (let i = 0; i < n; i++) start[i + 1] += start[i];
  const fill = start.slice(0, n);
  for (let e = 0; e < count; e++) {
    const k = fill[pairs[e * 2 + from]]++;
    list[k] = pairs[e * 2 + to];
    if (kind) kind[k] = kinds[e];
  }
  return { start, list, kind };
}
const later = adjacency(edges, 0, 1), earlier = adjacency(edges, 1, 0);

// The whole context graph: read edges (everything new in the room since the speaker last spoke there) and
// memory edges (the speaker's own previous message, any room). Contact distance counts hand-offs: a read
// costs 1, memory costs 0, so a meme an agent carries for weeks is still one hand-off from its source.
const contextPairs = contextBuf ? new Uint32Array(contextBuf) : edges;
const contextKinds = contextKindBuf ? new Uint8Array(contextKindBuf) : new Uint8Array(contextPairs.length / 2).fill(1);
const contextLater = adjacency(contextPairs, 0, 1, contextKinds), contextEarlier = adjacency(contextPairs, 1, 0, contextKinds);
const handoffs = new Int32Array(n).fill(0x7fffffff), deque = new Uint32Array(4 * n), touchedNodes = [];

// Exact hand-offs from i to each target, by 0-1 breadth-first search (memory edges to the front of the
// deque, reads to the back). Edges only run forward in time, so a path to an earlier target never passes
// through anything older than it: the search prunes there, and ends once every target is settled.
// A target missing from the result has no path at all.
function contactDistances(i, targets, memoryCost) {
  const result = new Map(), size = deque.length;
  for (const [adj, side] of [[contextEarlier, -1], [contextLater, 1]]) {
    const want = targets.filter(j => (daysOf(j) - daysOf(i)) * side >= 0 && j !== i);
    if (!want.length) continue;
    const bound = side < 0 ? Math.min(...want.map(daysOf)) : Math.max(...want.map(daysOf));
    const wanted = new Set(want);
    let head = 0, tail = 0;
    handoffs[i] = 0; touchedNodes.push(i); deque[tail++] = i;
    while (head !== tail && wanted.size) {
      const a = deque[head]; head = (head + 1) % size;
      if (wanted.delete(a)) result.set(a, side * handoffs[a]);
      for (let k = adj.start[a]; k < adj.start[a + 1]; k++) {
        const b = adj.list[k];
        if ((daysOf(b) - bound) * side > 0) continue;
        const w = adj.kind[k] === 0 ? memoryCost : 1, d = handoffs[a] + w;
        if (d >= handoffs[b]) continue;
        if (handoffs[b] === 0x7fffffff) touchedNodes.push(b);
        handoffs[b] = d;
        if (w === 0) { head = (head - 1 + size) % size; deque[head] = b; } else { deque[tail] = b; tail = (tail + 1) % size; }
      }
    }
    for (const t of touchedNodes) handoffs[t] = 0x7fffffff;
    touchedNodes.length = 0;
  }
  return result;
}

// cone per star: 0 the selected star, -h h hops into its past, +h h hops into its future, NONE outside.
// mark per star: 1 for the focused turn's direct connections (and itself), which stay lit in solo.
const NONE = 1e4, MAX_HOPS = 40;
const cone = new Float32Array(n).fill(NONE), mark = new Float32Array(n);
const coneAttr = new THREE.BufferAttribute(cone, 1), markAttr = new THREE.BufferAttribute(mark, 1);
geometry.setAttribute("cone", coneAttr);
geometry.setAttribute("mark", markAttr);
const hit = new Float32Array(n), hitAttr = new THREE.BufferAttribute(hit, 1);
geometry.setAttribute("hit", hitAttr);

const uniforms = { uTime: { value: 0 }, uPx: { value: 1 }, uSize: { value: 0.45 }, uFog: { value: 160 },
  uFocus: { value: 0 }, uReveal: { value: 0 }, uSolo: { value: 0 }, uDim: { value: 0.14 },
  uSearch: { value: 0 } };
const starMaterial = new THREE.ShaderMaterial({
  uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: /* glsl */`
    attribute vec3 color; attribute float reads, cone, mark, hit;
    uniform float uTime, uPx, uSize, uFog, uFocus, uReveal, uSolo, uDim, uSearch;
    varying vec3 vColor; varying float vBright;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      float h = fract(sin(dot(position.xy, vec2(12.9898, 78.233))) * 43758.5453);
      float twinkle = 0.8 + 0.2 * sin(uTime * (0.8 + 2.0 * h) + h * 40.0);
      float hops = abs(cone);
      float lit = step(hops, uReveal) * step(cone, 9000.0);                  // inside the cone and reached already
      float front = lit * exp(-pow((uReveal - hops) * 1.2, 2.0));           // the wavefront flares as it passes
      float marked = step(0.5, mark);
      float grow = mix(mix(1.0, mix(1.0, 1.0 + exp(-hops / 6.0), lit), uFocus), mix(0.8, 1.6, marked), uSolo);
      grow *= mix(1.0, mix(0.9, 1.7, hit), uSearch);
      float px = grow * uSize * (0.7 + reads / 10.0) * uPx / max(-mv.z, 0.1);
      float fog = 1.0 / (1.0 + pow(-mv.z / uFog, 2.0));   // the far past is a haze, not a pile-up
      fog = mix(fog, sqrt(fog), max(lit * uFocus, marked * uSolo));
      fog = mix(fog, 1.0, hit * uSearch);                     // matches shine through the haze
      float weight = lit * (0.12 + 1.6 * exp(-hops / 5.0)) + front * 1.5;       // near hops blaze, the far cone glows
      float cone_ = mix(1.0, mix(0.07, weight, lit), uFocus);
      float solo = mix(uDim * (1.0 + 2.2 * lit * exp(-hops / 5.0)), 1.8, marked);   // solo: connections, its cone, the sky
      float found = mix(mix(cone_, solo, uSolo), mix(0.16, 2.2, hit), uSearch * (1.0 - marked * uSolo));   // search
      vBright = found * twinkle * fog * mix(min(1.0, (px * px) / 9.0), 1.0, hit * uSearch);
      gl_PointSize = clamp(px, 3.0, 96.0);
      vec3 tint = cone < 0.0 ? vec3(1.0, 0.78, 0.45) : vec3(0.55, 0.8, 1.0);   // past warm, future cool
      vColor = mix(color, tint, 0.5 * lit * uFocus * step(0.5, hops) * (1.0 - uSolo));
      vColor = mix(vColor, vec3(dot(vColor, vec3(0.3, 0.55, 0.15))), 0.5 * max(uSolo * (1.0 - marked), uSearch * (1.0 - hit)));   // the rest, half grey
    }`,
  fragmentShader: /* glsl */`
    varying vec3 vColor; varying float vBright;
    void main() {
      vec2 c = gl_PointCoord * 2.0 - 1.0;
      float r2 = dot(c, c);
      if (r2 > 1.0) discard;
      float core = exp(-r2 * 30.0), halo = exp(-r2 * 5.0) * 0.3;
      gl_FragColor = vec4(mix(vColor, vec3(1.0), core * 0.7), (core + halo) * vBright);
    }`,
});
scene.add(new THREE.Points(geometry, starMaterial));

// lines: the speaker's previous message and the last few things they read, drawn faintly
const lineGeometry = new THREE.BufferGeometry();
lineGeometry.setAttribute("position", positionAttr);
lineGeometry.setAttribute("color", geometry.getAttribute("color"));
lineGeometry.setIndex(new THREE.BufferAttribute(new Uint32Array(edgeBuf), 1));
const lineUniforms = { uAlpha: { value: 0.05 } };
const lines = new THREE.LineSegments(lineGeometry, new THREE.ShaderMaterial({
  uniforms: lineUniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: `attribute vec3 color; varying vec3 vColor;
    void main() { vColor = color; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform float uAlpha; varying vec3 vColor; void main() { gl_FragColor = vec4(vColor, uAlpha); }`,
}));
lines.visible = false;
scene.add(lines);

// the cone's own edges: each one lights when the wavefront reaches its far end; gone in solo
const coneLineGeometry = new THREE.BufferGeometry();
coneLineGeometry.setAttribute("position", positionAttr);
coneLineGeometry.setAttribute("color", geometry.getAttribute("color"));
coneLineGeometry.setAttribute("cone", coneAttr);
coneLineGeometry.setIndex(new THREE.BufferAttribute(new Uint32Array(2), 1));
const coneLines = new THREE.LineSegments(coneLineGeometry, new THREE.ShaderMaterial({
  uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: `attribute vec3 color; attribute float cone; uniform float uReveal, uFocus, uSolo;
    varying vec3 vColor; varying float vAlpha;
    void main() {
      float hops = abs(cone);
      vAlpha = uFocus * (1.0 - 0.6 * uSolo) * step(hops, uReveal) * (0.006 + 0.35 * exp(-hops / 3.0));
      vColor = color;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`,
  fragmentShader: `varying vec3 vColor; varying float vAlpha; void main() { gl_FragColor = vec4(vColor, vAlpha); }`,
}));
coneLines.visible = false;
scene.add(coneLines);

// breadth-first from the selected star, MAX_HOPS each way; returns the turns reached, nearest first
function walk(origin, adj, sign) {
  const reached = [];
  let front = [origin];
  for (let hop = 1; hop <= MAX_HOPS && front.length; hop++) {
    const next = [];
    for (const i of front)
      for (let k = adj.start[i]; k < adj.start[i + 1]; k++) {
        const j = adj.list[k];
        if (cone[j] !== NONE) continue;
        cone[j] = sign * hop;
        next.push(j);
        reached.push(j);
      }
    front = next;
  }
  return reached;
}
let focusTarget = 0, soloTarget = 0;
function lightCone(i) {
  cone.fill(NONE);
  cone[i] = 0;
  const past = walk(i, earlier, -1), future = walk(i, later, 1);
  coneAttr.needsUpdate = true;
  // edges with both ends on the same side of the cone (or at its apex)
  const pairs = [];
  for (let e = 0; e < m; e++) {
    const a = cone[edges[e * 2]], b = cone[edges[e * 2 + 1]];
    if (a !== NONE && b !== NONE && a * b >= 0) pairs.push(edges[e * 2], edges[e * 2 + 1]);
  }
  coneLineGeometry.setIndex(new THREE.BufferAttribute(new Uint32Array(pairs), 1));
  coneLines.visible = true;
  focusTarget = 1;
  uniforms.uReveal.value = 0;
  return { past, future };
}

// echoes: the selected star's nearest neighbours in meaning, joined to it by a thread coloured by contact
const ECHO_K = 24, ECHO_MIN = 0.8;
const echoIndex = echoIndexBuf && new Uint32Array(echoIndexBuf), echoSim = echoSimBuf && new Uint8Array(echoSimBuf);
const CONTACT = { close: [1.0, 0.8, 0.35], near: [0.85, 0.85, 0.9], far: [0.45, 0.6, 0.85], none: [0.8, 0.45, 1.0],
  self: [0.4, 0.4, 0.45] };
const GROUPS = [["close", "in contact", "1 hand-off: one read the other, then or from memory"],
  ["near", "near", "2 to 3 hand-offs: through intermediaries"], ["far", "distant", "4 or more hand-offs"],
  ["none", "no contact", "no path through context: found independently"],
  ["self", "itself", "the same speaker, from its own memory"]];

// ribbons: quads expanded in screen space, so width is in pixels. They remember their star pairs, so they
// follow the stars while the lens blends.
const ribbonUniforms = { uResolution: { value: new THREE.Vector2(innerWidth, innerHeight) } };
const ribbonMaterial = new THREE.ShaderMaterial({
  uniforms: ribbonUniforms, transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: /* glsl */`
    attribute vec3 aStart, aEnd, aColor; attribute float aWidth;
    uniform vec2 uResolution;
    varying vec3 vColor; varying float vSide, vAlong;
    void main() {
      vec4 a = projectionMatrix * modelViewMatrix * vec4(aStart, 1.0);
      vec4 b = projectionMatrix * modelViewMatrix * vec4(aEnd, 1.0);
      a.w = max(a.w, 0.01); b.w = max(b.w, 0.01);
      vec2 dir = normalize((b.xy / b.w - a.xy / a.w) * uResolution + 1e-6);
      vec4 p = mix(a, b, position.x);
      p.xy += vec2(-dir.y, dir.x) * position.y * (aWidth + 2.0) / uResolution * p.w;   // 2 px of soft edge
      gl_Position = p;
      vColor = aColor; vSide = position.y * (aWidth + 2.0) / max(aWidth, 1.0); vAlong = position.x;
    }`,
  fragmentShader: /* glsl */`
    varying vec3 vColor; varying float vSide, vAlong;
    void main() {
      float edge = clamp(1.0 - (abs(vSide) - 1.0) * 2.0, 0.0, 1.0);        // full across the width, soft outside
      gl_FragColor = vec4(vColor, edge * mix(0.35, 1.0, vAlong));           // dimmer at the start
    }`,
});
const ribbonSets = [];
function makeRibbons(capacity) {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array([0, -1, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0]), 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  const r = { start: new Float32Array(capacity * 3), end: new Float32Array(capacity * 3),
    color: new Float32Array(capacity * 3), width: new Float32Array(capacity), pairs: [] };
  for (const [name, array, size] of [["aStart", r.start, 3], ["aEnd", r.end, 3], ["aColor", r.color, 3], ["aWidth", r.width, 1]])
    geometry.setAttribute(name, new THREE.InstancedBufferAttribute(array, size));
  r.mesh = new THREE.Mesh(geometry, ribbonMaterial);
  r.mesh.frustumCulled = false;
  r.mesh.visible = false;
  r.set = (k, a, b, color, width) => {        // from star a to star b
    r.pairs[k] = [a, b];
    r.color.set(color, k * 3);
    r.width[k] = width;
  };
  r.show = count => {
    r.pairs.length = count;
    geometry.instanceCount = count;
    r.mesh.visible = count > 0;
    r.follow();
    for (const a of Object.values(geometry.attributes)) a.needsUpdate = true;
  };
  r.follow = () => {
    r.pairs.forEach(([a, b], k) => {
      r.start.set(position.subarray(a * 3, a * 3 + 3), k * 3);
      r.end.set(position.subarray(b * 3, b * 3 + 3), k * 3);
    });
    geometry.attributes.aStart.needsUpdate = geometry.attributes.aEnd.needsUpdate = true;
  };
  scene.add(r.mesh);
  ribbonSets.push(r);
  return r;
}
const echoRibbons = makeRibbons(ECHO_K), contextRibbons = makeRibbons(32);

// How strongly could j have been carried to (or from) i? Similarity above 0.8, times closeness through
// fresh context (hops with no memory): 1 hop is full strength, falling by e every 3 hops. Reachable only
// through an agent's memory: a faint floor. No path at all, or the same speaker: a hairline.
function strengthOf(e) {
  if (e.contact === "self" || e.contact === "none") return 0;
  const similar = Math.min(1, Math.max(0, (e.sim - 0.8) / 0.2));
  const close = e.hops === undefined ? 0.12 : Math.max(0.12, Math.exp(-(Math.abs(e.hops) - 1) / 3));
  return (0.35 + 0.65 * similar) * close;
}
function contactOf(i, j, hops) {
  if (rgbs[i * 4 + 3] === rgbs[j * 4 + 3]) return "self";
  if (hops === undefined) return "none";
  const h = Math.abs(hops);
  return h <= 1 ? "close" : h <= 3 ? "near" : "far";
}
function echoesOf(i) {
  if (!echoIndex) return [];
  const list = [];
  for (let k = 0; k < ECHO_K; k++) {
    const j = echoIndex[i * ECHO_K + k], sim = echoSim[i * ECHO_K + k] / 255;
    if (sim < ECHO_MIN) break;
    list.push({ j, sim });
  }
  const other = list.filter(e => rgbs[e.j * 4 + 3] !== rgbs[i * 4 + 3]).map(e => e.j);
  const handoffMap = contactDistances(i, other, 0), hopMap = contactDistances(i, other, 1);
  for (const e of list) {
    e.handoffs = handoffMap.get(e.j); e.hops = hopMap.get(e.j);
    e.contact = contactOf(i, e.j, e.handoffs);
    e.strength = strengthOf(e);
  }
  list.forEach((e, k) => {
    const bright = 0.25 + 0.95 * e.strength;
    echoRibbons.set(k, i, e.j, CONTACT[e.contact].map(v => v * bright * 0.8), 1 + 5 * e.strength);
  });
  echoRibbons.show(list.length);
  return list.sort((a, b) => daysOf(a.j) - daysOf(b.j));
}

// what a turn read (latest direct inputs), its own previous message, and who read it (first readers)
const INPUTS = 5, READERS = 5, ECHO_LABELS = 6;
function contextOf(i) {
  const inputs = [], readers = [];
  let memory = -1;
  for (let k = contextEarlier.start[i]; k < contextEarlier.start[i + 1]; k++)
    if (contextEarlier.kind[k]) inputs.push(contextEarlier.list[k]); else memory = contextEarlier.list[k];
  for (let k = contextLater.start[i]; k < contextLater.start[i + 1]; k++)
    if (contextLater.kind[k]) readers.push(contextLater.list[k]);
  const byTime = (a, b) => daysOf(a) - daysOf(b);
  return { inputs: inputs.sort(byTime).slice(-INPUTS), readers: readers.sort(byTime).slice(0, READERS), memory };
}

// camera: position plus yaw and pitch; wheel flies along the view, left drag looks, right drag pans
const cam = { x: 0, y: 0, z: 80, yaw: 0, pitch: 0 };
const params = new URLSearchParams(location.search);
let autopilot = !params.has("cam") && !params.has("open"), lastInput = params.has("cam") ? Infinity : -1e9, pilotTime = 0;
if (params.has("cam")) [cam.x, cam.y, cam.z, cam.yaw, cam.pitch] = params.get("cam").split(",").map(Number);
if (params.has("lines")) lines.visible = true;
if (params.has("clean")) document.body.classList.add("hide-ui");
function forward(c = cam) {
  return new THREE.Vector3(-Math.sin(c.yaw) * Math.cos(c.pitch), Math.sin(c.pitch), -Math.cos(c.yaw) * Math.cos(c.pitch));
}
function applyCamera() {
  camera.position.set(cam.x, cam.y, cam.z);
  camera.lookAt(camera.position.clone().add(forward()));
}
let camGoal = null;
function touched() { autopilot = false; lastInput = performance.now(); camGoal = null; }
// a pose at offset (dx, dy, dz) from world point p, looking at p
function poseLookingAt(p, dx, dy, dz) {
  const f = new THREE.Vector3(-dx, -dy, -dz).normalize();
  return { x: p[0] + dx, y: p[1] + dy, z: p[2] + dz, yaw: Math.atan2(-f.x, -f.z), pitch: Math.asin(f.y) };
}

const canvas = renderer.domElement;
let drag = null;
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, button: e.button, moved: 0 }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", e => {
  if (!drag) { hover(e.clientX, e.clientY); return; }
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.moved += Math.abs(dx) + Math.abs(dy);
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.moved < 4) return;
  touched();
  if (drag.button === 0) {
    cam.yaw += dx * 0.003;
    cam.pitch = Math.max(-1.4, Math.min(1.4, cam.pitch - dy * 0.003));
  } else {
    const k = 0.0015 * (selected >= 0 ? 12 : Math.max(20, Math.abs(cam.z) * 0.3 + 40));
    const right = new THREE.Vector3().crossVectors(forward(), new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3().crossVectors(right, forward());
    cam.x += (-dx * right.x + dy * up.x) * k; cam.y += (-dx * right.y + dy * up.y) * k; cam.z += (-dx * right.z + dy * up.z) * k;
  }
});
canvas.addEventListener("pointerup", e => { if (drag && drag.moved < 4 && drag.button === 0) { touched(); pick(e.clientX, e.clientY); } drag = null; });
canvas.addEventListener("wheel", e => {
  e.preventDefault(); touched();
  const f = forward(), step = -e.deltaY * (selected >= 0 ? 0.02 : 0.08 * Math.max(1, Math.abs(cam.z) / 60));
  cam.x += f.x * step; cam.y += f.y * step; cam.z += f.z * step;
}, { passive: false });

addEventListener("keydown", e => {
  if (e.target === searchBox) return;
  const k = e.key.toLowerCase();
  if (k === "/") { e.preventDefault(); searchBox.focus(); searchBox.select(); return; }
  if (k === "l") lines.visible = !lines.visible;
  else if (k === "a") { autopilot = !autopilot; pilotTime = 0; }
  else if (k === "h") document.body.classList.toggle("hide-ui");
  else if (k === "i") panel.style.display = panel.style.display === "block" ? "none" : selected >= 0 ? "block" : "none";
  else if (k === "escape") leave();
  else if (k === "backspace") { e.preventDefault(); if (trail.length) focus(trail.pop(), false); else leave(); }
  else if (k === "arrowright" || k === "arrowleft") { e.preventDefault(); cycle(k === "arrowright" ? 1 : -1); }
  else if ((k === "enter" || k === " ") && cursor >= 0) { e.preventDefault(); focus(labels[cursor].j); }
  else if (k === "f" && selected >= 0 && zoomed) turnAround();
  else if (k === "z") toggleZoom();
  else if ((k === "[" || k === "]") && selected < 0) { tau *= k === "]" ? 1.5 : 1 / 1.5; setLens(overviewLens()); }
  else if (k === "[" || k === "]") uniforms.uDim.value = Math.min(1, Math.max(0.02, uniforms.uDim.value * (k === "]" ? 1.4 : 1 / 1.4)));
  else if (k === "=" || k === "-") uniforms.uSize.value *= k === "=" ? 1.2 : 1 / 1.2;
  else if (k === "." || k === ",") lineUniforms.uAlpha.value *= k === "." ? 1.5 : 1 / 1.5;
});

// picking: project every star, take the nearest to the pointer within 14 px, preferring the closest to the camera
const panel = document.getElementById("panel"), shards = new Map();
const viewProjection = new THREE.Matrix4();
function pick(sx, sy) {
  const hit = labelAt(sx, sy);
  if (hit) { focus(hit.j); return; }
  viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const e = viewProjection.elements, w2 = innerWidth / 2, h2 = innerHeight / 2;
  let best = -1, bestScore = 14 * 14;
  for (let i = 0; i < n; i++) {
    const x = position[i * 3], y = position[i * 3 + 1], z = position[i * 3 + 2];
    const w = e[3] * x + e[7] * y + e[11] * z + e[15];
    if (w <= 0.1) continue;
    const px = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / w + 1) * w2 - sx;
    const py = (1 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / w) * h2 - sy;
    const score = px * px + py * py + w * 0.02 - (mark[i] > 0.5 ? 60 : 0);    // connections win close calls
    if (score < bestScore) { bestScore = score; best = i; }
  }
  if (best >= 0) focus(best);
}

// Search streams through the text shards (fetched once, then cached), lighting matches as each shard
// arrives: case-insensitive, every word must appear. Enter in the box opens the first match; Esc clears.
const searchBox = document.getElementById("search"), searchCount = document.getElementById("found");
let searchRun = 0, searchTarget = 0, hits = [];
const shardOf = s => {
  if (!shards.has(s)) shards.set(s, fetch(`data/text/${String(s).padStart(4, "0")}.json`).then(r => r.json()));
  return shards.get(s);
};
async function search(query) {
  const run = ++searchRun, words = query.toLowerCase().split(/\s+/).filter(Boolean);
  hit.fill(0); hitAttr.needsUpdate = true; hits = [];
  searchTarget = words.length ? 1 : 0;
  searchCount.textContent = "";
  if (!words.length) return;
  const count = Math.ceil(n / meta.shard);
  let next = 0, done = 0;
  const worker = async () => {
    while (next < count && run === searchRun) {
      const s = next++, rows = await shardOf(s);
      if (run !== searchRun) return;
      rows.forEach((r, k) => {
        const text = r.lowerText ??= `${r.speaker} ${r.text}`.toLowerCase();
        if (words.every(w => text.includes(w))) { hit[s * meta.shard + k] = 1; hits.push(s * meta.shard + k); }
      });
      hitAttr.needsUpdate = true;
      done++;
      searchCount.textContent = `${hits.length.toLocaleString()} found` + (done < count ? ` · ${Math.round(100 * done / count)}%` : "");
    }
  };
  await Promise.all([worker(), worker(), worker(), worker()]);
}
let searchTimer = 0;
searchBox.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => search(searchBox.value), 250); });
searchBox.addEventListener("keydown", e => {
  if (e.key === "Escape") { searchBox.value = ""; search(""); searchBox.blur(); }
  else if (e.key === "Enter" && hits.length) { searchBox.blur(); focus(Math.min(...hits)); }
});
if (params.has("q")) { searchBox.value = params.get("q"); search(searchBox.value); }

const textOf = async j => {
  const s = Math.floor(j / meta.shard);
  if (!shards.has(s)) shards.set(s, fetch(`data/text/${String(s).padStart(4, "0")}.json`).then(r => r.json()));
  return (await shards.get(s))[j % meta.shard];
};
const rgbOf = c => `rgb(${c.map(v => Math.round(v * 255)).join(",")})`;
const speakerColor = j => meta.speakers[rgbs[j * 4 + 3]].color;
function ago(i, j) {
  const seconds = Math.round((daysOf(j) - daysOf(i)) * 86400), a = Math.abs(seconds);
  const text = a < 120 ? `${a} s` : a < 5400 ? `${Math.round(a / 60)} min` : a < 172800 ? `${Math.round(a / 3600)} h`
    : `${Math.round(a / 86400)} d`;
  return (seconds < 0 ? "−" : "+") + text;
}

// Focus: fly into turn i. The lens re-anchors at its moment, the camera glides to just in front of it, the
// star opens into its card, and only its connections stay lit, each labelled where its star is.
let selected = -1, overviewPose = null;
const trail = [];
let labels = [], cursor = -1, facing = 1;          // facing 1 looks into the past, -1 into the future

function focus(i, remember = true) {
  if (selected < 0) overviewPose = { ...cam };
  else if (remember && selected !== i) trail.push(selected);
  autopilot = false; lastInput = performance.now();
  selected = i;
  facing = 1;
  const ctx = contextOf(i);
  focusCtx = ctx;
  if (zoomed) zoomInto(i, ctx); else { setLens(overviewLens(), params.has("reveal")); }
  const { past, future } = lightCone(i);
  const echoes = echoesOf(i);
  if (!zoomed) camGoal = approach(i);
  if (params.has("reveal")) Object.assign(cam, camGoal);
  soloTarget = 1;
  if (params.has("reveal")) { uniforms.uFocus.value = uniforms.uSolo.value = 1; uniforms.uReveal.value = MAX_HOPS + 2; }

  // connections: marked stars, warm threads in, cool threads out, a label on each
  mark.fill(0);
  mark[i] = 1;
  const items = [];
  if (ctx.memory >= 0) items.push({ j: ctx.memory, kind: "memory", tag: "its previous" });
  for (const j of ctx.inputs) items.push({ j, kind: "input", tag: "it read" });
  for (const j of ctx.readers) items.push({ j, kind: "reader", tag: "read it" });
  const seen = new Set(items.map(t => t.j));
  echoes.filter(e => !seen.has(e.j) && e.contact !== "self").sort((a, b) => b.strength - a.strength || b.sim - a.sim)
    .slice(0, ECHO_LABELS).forEach(e => items.push({ j: e.j, kind: "echo", tag: `echo ${e.sim.toFixed(2)}`, contact: e.contact }));
  items.forEach(t => { mark[t.j] = 1; });
  markAttr.needsUpdate = true;
  const warm = [1.0, 0.75, 0.45], cool = [0.5, 0.75, 1.0], dim = [0.45, 0.45, 0.5];
  const threads = items.filter(t => t.kind !== "echo");
  threads.forEach((t, k) => contextRibbons.set(k, t.kind === "reader" ? i : t.j, t.kind === "reader" ? t.j : i,
    (t.kind === "reader" ? cool : t.kind === "memory" ? dim : warm).map(v => v * 0.7), 2));
  contextRibbons.show(threads.length);
  buildLabels(i, items);
  openCard(i);
  fillPanel(i, past, future, echoes);
}

function leave() {
  if (selected < 0) return;
  selected = -1;
  zoomed = false;
  trail.length = 0;
  soloTarget = 0; focusTarget = 0;
  setLens(overviewLens());
  if (overviewPose) camGoal = overviewPose;
  for (const r of ribbonSets) r.show(0);
  mark.fill(0); markAttr.needsUpdate = true;
  clearLabels();
  closeCard();
  panel.style.display = "none";
}

// Framing. From the side, time runs across the screen (past left, future right) centred on the turn, and
// meaning runs up and down; the distance fits what it read and who read it. F toggles to looking down the
// time axis into its past.
let view = "side", focusDistance = 20;
function frameFor(i, ctx, l) {
  let across = 6, high = 3;
  for (const j of [...ctx.inputs, ...ctx.readers]) {
    across = Math.max(across, Math.min(60, Math.abs(depthIn(l, daysOf(j)))));
    high = Math.max(high, Math.min(40, Math.abs(yIn(l, j) - l.centre[1])));
  }
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2), tanH = tanV * camera.aspect;
  return Math.max(16, (across + 2) / (tanH * 0.8), (high + 6) / (tanV * 0.55));
}
function viewPose(i) {
  const p = [lens.centre[0], lens.centre[1], 0], d = focusDistance;
  return view === "side" ? poseLookingAt(p, -d * 0.97, d * 0.1, d * 0.18) : poseLookingAt(p, -4, 1.5, 13);
}
// In the sky: turn to face the star and come partway closer, leaving the sky around it.
let zoomed = false, focusCtx = null;
function approach(i) {
  const l = overviewLens(), star = new THREE.Vector3(stars[i * 4] * SPREAD, stars[i * 4 + 1] * SPREAD, depthIn(l, daysOf(i)));
  const from = new THREE.Vector3(cam.x, cam.y, cam.z), away = from.clone().sub(star);
  const d = away.length() || 1, keep = Math.min(90, Math.max(32, d * 0.55));
  if (away.z < 0) away.set(away.x, away.y, Math.abs(away.z) + 1);          // stay on the present's side of it
  away.normalize();
  focusDistance = keep;
  return poseLookingAt(star.toArray(), away.x * keep, away.y * keep, away.z * keep);
}
// Z: zoom into its moment. Time re-anchors at the turn with tau fitted to its conversation, meaning squashes
// toward it, and the camera looks from the side: what it read on the left, who read it on the right.
function zoomInto(i, ctx) {
  const focused = focusLens(i, [...ctx.inputs, ...ctx.readers]);
  setLens(focused, params.has("reveal"));
  focusDistance = frameFor(i, ctx, focused);
  camGoal = viewPose(i);
}
function toggleZoom() {
  if (selected < 0) return;
  zoomed = !zoomed;
  if (zoomed) zoomInto(selected, focusCtx);
  else { setLens(overviewLens()); camGoal = approach(selected); }
  openCard(selected);                      // resized for the new distance
}
function turnAround() {
  view = view === "side" ? "past" : "side";
  camGoal = viewPose(selected);
}

// Text in the world. A card is a camera-facing quad drawn by the loom's node shader: a rounded box by signed
// distance, dark fill, a fine edge line and a glow in the model's colour, and its text from a canvas texture.
// Output is premultiplied, so the fill covers the stars behind and the edge adds light; fog is the stars'.
// The focus card is sized in world units and hangs off its star (its right edge at the star); opening grows
// the box out of the star point, a circle widening into the box, and the text arrives last. Labels are sized
// in pixels so they read at any distance.
const cardUniforms = { uResolution: ribbonUniforms.uResolution, uFog: uniforms.uFog };
const cardMaterial = (map, o) => new THREE.ShaderMaterial({
  transparent: true, depthTest: false, depthWrite: false,
  blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
  uniforms: { ...cardUniforms, uMap: { value: map }, uCenter: { value: new THREE.Vector3() }, uSize: { value: new THREE.Vector2() },
    uOffset: { value: new THREE.Vector2() }, uAnchor: { value: new THREE.Vector2(...(o.anchor || [0, 0])) },
    uMargin: { value: o.margin }, uRadius: { value: o.radius },
    uLine: { value: o.line }, uGlow: { value: o.glow }, uFill: { value: o.fill }, uColor: { value: new THREE.Color(...o.color) },
    uOpen: { value: o.open ?? 0 }, uAlpha: { value: 0 }, uPixel: { value: o.pixel ? 1 : 0 } },
  vertexShader: /* glsl */`
    uniform vec3 uCenter; uniform vec2 uSize, uOffset, uResolution; uniform float uPixel;
    varying vec2 vP; varying float vDepth;
    void main() {
      vec2 local = position.xy * uSize + uOffset;             // quad units: world, or pixels when uPixel
      vP = position.xy * uSize;
      vec4 c = modelViewMatrix * vec4(uCenter, 1.0);
      vDepth = -c.z;
      if (uPixel > 0.5) {
        vec4 clip = projectionMatrix * c;
        clip.xy += local * 2.0 / uResolution * clip.w;
        gl_Position = clip;
      } else {
        c.xy += local;
        gl_Position = projectionMatrix * c;
      }
    }`,
  fragmentShader: /* glsl */`
    uniform sampler2D uMap; uniform vec2 uSize, uAnchor; uniform vec3 uColor;
    uniform float uMargin, uRadius, uLine, uGlow, uFill, uOpen, uAlpha, uFog, uPixel;
    varying vec2 vP; varying float vDepth;
    float box(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
    void main() {
      vec2 half_ = uSize * 0.5 - uMargin;
      float e = uOpen * uOpen * (3.0 - 2.0 * uOpen);
      vec2 b = mix(vec2(uRadius), half_, e);                   // grows from a circle at the star, on its anchor edge
      vec2 centre = uAnchor * (half_ - b);
      float d = box(vP - centre, b, min(uRadius, min(b.x, b.y)));
      float aa = fwidth(d);
      float inside = 1.0 - smoothstep(-aa, aa, d);
      float window_ = 1.0 - smoothstep(0.55 * uMargin, uMargin, d);        // glow reaches zero before the quad edge
      float edge = exp(-abs(d) / uLine) * 0.9 + exp(-max(d, 0.0) / uGlow) * 0.35 * (1.0 - inside) * window_;
      float fog = uPixel > 0.5 ? 1.0 : 1.0 / (1.0 + pow(vDepth / uFog, 2.0));
      float a = uFill * inside;
      vec3 rgb = vec3(0.03, 0.03, 0.045) * a + uColor * edge;
      vec2 uv = (vP + half_) / (2.0 * half_);
      vec4 t = texture2D(uMap, uv);
      float text = smoothstep(0.75, 1.0, uOpen) * inside * t.a;
      rgb += t.rgb * text;
      a = max(a, text);
      gl_FragColor = vec4(rgb, a) * uAlpha * fog;
    }`,
});
const quad = new THREE.PlaneGeometry(1, 1);

// draw text into a canvas at 2x: a header line in the model's colour, a dim meta line, then wrapped body
const INK = "#e8e4dc", DIM = "#8a8478", FONT = 'ui-monospace, "Cascadia Mono", Consolas, monospace';
function wrap(ctx, text, width) {
  const out = [];
  for (const para of text.split("\n")) {
    let line = "";
    for (const word of para.split(/(\s+)/)) {
      if (ctx.measureText(line + word).width <= width) { line += word; continue; }
      if (line.trim()) out.push(line.trimEnd());
      line = word.trimStart();
      while (ctx.measureText(line).width > width) {                  // a token longer than the line
        let k = line.length;
        while (k > 1 && ctx.measureText(line.slice(0, k)).width > width) k--;
        out.push(line.slice(0, k)); line = line.slice(k);
      }
    }
    out.push(line.trimEnd());
  }
  return out;
}
function textCanvas({ head, headColor, meta, body, width, maxLines, pad = 28 }) {
  const c = document.createElement("canvas"), ctx = c.getContext("2d");
  const big = `600 28px ${FONT}`, small = `22px ${FONT}`, text = `25px ${FONT}`;
  ctx.font = big;
  const inner = width ? width - 2 * pad : Math.ceil(ctx.measureText(head).width);
  ctx.font = text;
  let lines = body ? wrap(ctx, body, inner) : [];
  if (lines.length > maxLines) { lines = lines.slice(0, maxLines); lines[maxLines - 1] += " …"; }
  c.width = inner + 2 * pad;
  c.height = pad * 2 + 34 + (meta ? 32 : 0) + (lines.length ? 12 + lines.length * 34 : 0);
  let y = pad + 26;
  ctx.font = big; ctx.fillStyle = headColor; ctx.fillText(head, pad, y);
  if (meta) { y += 32; ctx.font = small; ctx.fillStyle = DIM; ctx.fillText(meta, pad, y); }
  if (lines.length) { y += 12; ctx.font = text; ctx.fillStyle = INK; for (const l of lines) { y += 34; ctx.fillText(l, pad, y); } }
  const texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return { texture, w: c.width / 2, h: c.height / 2 };          // css pixels
}

// the focus card
const CARD_PX = 460;
let card = null;
function closeCard() { if (card) { card.closing = true; } }
function dropCard(c) { scene.remove(c.mesh); c.mesh.material.dispose(); c.texture.dispose(); }
async function openCard(i) {
  const t = await textOf(i);
  if (selected !== i) return;
  if (card) dropCard(card);
  const color = speakerColor(i);
  const { texture, w, h } = textCanvas({ head: t.speaker, headColor: rgbOf(color), meta: `${t.time} UTC · #${t.room}`,
    body: t.text, width: 920, maxLines: 22 });
  const unit = focusDistance / (renderer.domElement.clientHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)));
  const width = CARD_PX * unit, height = width * h / w, margin = 18 * unit, gap = 10 * unit;
  const mesh = new THREE.Mesh(quad, cardMaterial(texture, { margin, radius: 9 * unit, line: 1.2 * unit, glow: 8 * unit,
    fill: 0.9, color, anchor: [0, 1] }));
  mesh.frustumCulled = false;
  mesh.renderOrder = 10;
  const u = mesh.material.uniforms;
  u.uSize.value.set(width + 2 * margin, height + 2 * margin);
  u.uOffset.value.set(0, -height / 2 - gap);               // box top edge just below the star
  scene.add(mesh);
  card = { mesh, texture, j: i, width, height, open: params.has("reveal") ? 1 : 0, closing: false };
}

// labels on the connections, at their own stars: a header by default; the cursor's label shows its text
function clearLabels() { for (const l of labels) dropCard(l); labels = []; cursor = -1; }
function labelTexture(l, full) {
  const { texture, w, h } = textCanvas({ head: l.head, headColor: rgbOf(speakerColor(l.j).map(v => 0.45 + 0.55 * v)),
    body: full ? l.text : "", width: full ? 680 : 0, maxLines: 9, pad: 16 });
  if (l.texture) l.texture.dispose();
  Object.assign(l, { texture, w, h, full });
  const u = l.mesh.material.uniforms;
  u.uMap.value = texture;
  u.uSize.value.set(w + 2 * 8, h + 2 * 8);
  u.uOffset.value.set(w / 2 + 18, 0);                      // left edge 18 px right of the star
}
function buildLabels(i, items) {
  clearLabels();
  items.sort((a, b) => daysOf(a.j) - daysOf(b.j));
  for (const t of items) {
    const color = t.contact ? CONTACT[t.contact] : speakerColor(t.j);
    const mesh = new THREE.Mesh(quad, cardMaterial(null, { margin: 8, radius: 7, line: 1.1, glow: 5, fill: 0.78,
      color: color.map(v => v * 0.8), open: 1, pixel: true }));
    mesh.frustumCulled = false;
    mesh.renderOrder = 11;
    mesh.visible = false;
    scene.add(mesh);
    const l = { ...t, mesh, head: "", text: "", alpha: 0 };
    labels.push(l);
    textOf(t.j).then(x => {
      if (!labels.includes(l)) return;
      l.head = `${x.speaker} · ${ago(i, t.j)} · ${t.tag}`;
      l.text = x.text;
      labelTexture(l, false);
    });
  }
}
function cycle(step) {
  if (labels.length) cursor = (cursor + step + labels.length) % labels.length;
}
function labelAt(x, y) {
  return labels.find(l => l.rect && x >= l.rect[0] && x <= l.rect[2] && y >= l.rect[1] && y <= l.rect[3]);
}
function hover(x, y) {
  const l = labelAt(x, y);
  canvas.style.cursor = l ? "pointer" : "";
  if (l) cursor = labels.indexOf(l);
}

const projected = new THREE.Vector3();
function screenOf(j) {
  projected.fromArray(position, j * 3);
  const distance = projected.distanceTo(camera.position);
  projected.project(camera);
  if (projected.z > 1 || Math.abs(projected.x) > 1.3 || Math.abs(projected.y) > 1.3) return null;
  return { x: (projected.x + 1) / 2 * innerWidth, y: (1 - projected.y) / 2 * innerHeight, distance };
}
// each frame: the card opens (or closes), follows its star; labels go nearest first, and one that would overlap
// a label already placed, or the card, is left out (its star stays lit). The cursor's label always shows.
function placeOverlays(dt) {
  const pxPerUnit = renderer.domElement.clientHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
  const placed = [];
  if (card) {
    card.open = Math.max(0, Math.min(1, card.open + (card.closing ? -dt * 3 : dt * 1.4)));
    const u = card.mesh.material.uniforms;
    u.uCenter.value.fromArray(position, card.j * 3);
    u.uOpen.value = card.open;
    u.uAlpha.value = card.closing ? card.open : 1;
    if (card.closing && card.open <= 0) { dropCard(card); card = null; }
    else {
      const s = screenOf(card.j);
      if (s && card.open > 0.5) {
        const k = pxPerUnit / s.distance;
        placed.push([s.x - card.width * k / 2, s.y, s.x + card.width * k / 2, s.y + card.height * k + 16]);
      }
    }
  }
  const order = labels.map((l, k) => ({ l, k, p: l.texture && screenOf(l.j) })).filter(o => o.p)
    .sort((a, b) => (b.k === cursor) - (a.k === cursor) || a.p.distance - b.p.distance);
  for (const l of labels) { l.show = false; l.rect = null; }
  for (const { l, k, p } of order) {
    const current = k === cursor;
    if (l.full !== current) labelTexture(l, current);
    const r = [p.x + 18, p.y - l.h / 2, p.x + 18 + l.w, p.y + l.h / 2];
    if (!current && placed.some(q => r[0] < q[2] && r[2] > q[0] && r[1] < q[3] && r[3] > q[1])) continue;
    placed.push(r);
    l.show = true; l.rect = r;
    l.mesh.renderOrder = current ? 12 : 11;
    const u = l.mesh.material.uniforms;
    u.uCenter.value.fromArray(position, l.j * 3);
    u.uColor.value.setRGB(...(current ? [1, 1, 1] : (l.contact ? CONTACT[l.contact] : speakerColor(l.j)).map(v => v * 0.8)));
  }
  for (const l of labels) {
    l.alpha = params.has("reveal") ? +l.show : Math.max(0, Math.min(1, l.alpha + (l.show ? dt * 4 : -dt * 6)));
    l.mesh.visible = l.alpha > 0;
    l.mesh.material.uniforms.uAlpha.value = l.alpha * (selected >= 0 ? Math.min(1, Math.max(0.8, 40 / (screenOf(l.j)?.distance || 40))) : 1);
  }
}

function fillPanel(i, past, future, echoes) {
  textOf(i).then(t => {
    if (selected !== i) return;
    panel.innerHTML = "";
    const who = Object.assign(document.createElement("div"), { className: "who", textContent: t.speaker });
    who.style.color = rgbOf(speakerColor(i));
    panel.append(who,
      Object.assign(document.createElement("div"), { className: "meta",
        textContent: `${t.time} UTC · #${t.room} · read by ${reads[i]} later turns` }),
      Object.assign(document.createElement("div"), { className: "text", textContent: t.text }),
      coneSummary("past cone", "could have shaped it", past, "back to", "#ffc773"),
      coneSummary("future cone", "it could have reached", future, "out to", "#8cccff"),
      echoList(i, echoes));
    panel.scrollTop = 0;
  });
}

// the echoes, grouped by hand-offs, in time order
function echoList(i, echoes) {
  const box = Object.assign(document.createElement("div"), { className: "echoes" });
  if (!echoes.length) { box.textContent = echoIndex ? "no echoes: nothing else this similar (0.80)" : ""; return box; }
  box.append(Object.assign(document.createElement("div"), { className: "head",
    textContent: `echoes · ${echoes.length} messages similar in meaning, grouped by hand-offs between them` }));
  for (const [key, name, gloss] of GROUPS) {
    const members = echoes.filter(e => e.contact === key);
    if (!members.length) continue;
    const group = document.createElement("details");
    group.className = "group";
    group.open = key !== "self";
    const summary = document.createElement("summary");
    summary.innerHTML = `<span class="count"></span> <span class="name"></span> <span class="gloss"></span>`;
    summary.querySelector(".count").textContent = members.length;
    summary.querySelector(".name").textContent = name;
    summary.querySelector(".name").style.color = rgbOf(CONTACT[key]);
    summary.querySelector(".gloss").textContent = gloss;
    group.append(summary);
    for (const { j, sim, hops, handoffs: h0 } of members) {
      const item = Object.assign(document.createElement("div"), { className: "echo" });
      const when = daysOf(j) < daysOf(i) ? "earlier" : "later";
      const h = Math.abs(h0), distance = h0 === undefined ? "" : (h === 0 ? " · same thread"
        : ` · ${h} hand-off${h > 1 ? "s" : ""}`) + (hops === undefined ? "" : ` · ${Math.abs(hops)} hops fresh`);
      const line = Object.assign(document.createElement("div"), { className: "line" });
      const snip = Object.assign(document.createElement("div"), { className: "snip" });
      item.append(line, snip);
      item.addEventListener("click", () => focus(j));
      group.append(item);
      textOf(j).then(t => {
        line.textContent = `${sim.toFixed(2)} · ${t.speaker} · ${t.time.slice(0, 16)} · ${when}${distance}`;
        snip.textContent = t.text.length > 180 ? t.text.slice(0, 180) + "…" : t.text;
      });
    }
    box.append(group);
  }
  return box;
}

// who is in a cone: turns, speakers by count, and how far in time it reaches
function coneSummary(title, verb, list, reach, tint) {
  const box = Object.assign(document.createElement("div"), { className: "cone" });
  box.style.borderColor = tint;
  if (!list.length) { box.textContent = `${title}: empty`; return box; }
  const counts = new Map();
  let far = list[0];
  for (const j of list) {
    const k = rgbs[j * 4 + 3];
    counts.set(k, (counts.get(k) || 0) + 1);
    if ((daysOf(j) - daysOf(far)) * cone[j] > 0) far = j;     // earliest in the past, latest in the future
  }
  const date = j => new Date(t0 + daysOf(j) * 86400000).toISOString().slice(0, 16).replace("T", " ");
  const top = [...counts].sort((a, b) => b[1] - a[1]).slice(0, 6)
    .map(([k, c]) => `${meta.speakers[k].name} ${c}`).join(" · ");
  const head = Object.assign(document.createElement("div"), { className: "head",
    textContent: `${title} · ${list.length.toLocaleString()} turns · ${counts.size} speakers · ${reach} ${date(far)}` });
  head.style.color = tint;
  box.append(head, Object.assign(document.createElement("div"), { textContent: `${verb}: ${top}` }));
  return box;
}

addEventListener("resize", () => {
  ribbonUniforms.uResolution.value.set(innerWidth, innerHeight);
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
});
if (params.has("open")) { zoomed = params.has("zoom"); focus(Number(params.get("open"))); }

const when = document.getElementById("when");
const ease = (value, target, dt, rate) => value + (target - value) * (1 - Math.exp(-dt * rate));
let previous = performance.now();
renderer.setAnimationLoop(now => {
  const dt = Math.min(0.1, (now - previous) / 1000);
  previous = now;
  if (!autopilot && selected < 0 && now - lastInput > 45000 && lastInput !== Infinity) { autopilot = true; pilotTime = 0; }
  if (autopilot) {
    // drift back through time and return, slowly, wandering a little across the map; glide into the path
    pilotTime += dt;
    const p = pilotTime * 0.012, rate = Math.min(2, 0.05 + pilotTime * 0.05);
    const goal = { z: 80 - DEPTH * 0.55 * (1 - Math.cos(p)), x: 25 * Math.sin(p * 1.7), y: 15 * Math.sin(p * 1.1),
      yaw: 0.12 * Math.sin(p * 1.3), pitch: 0.08 * Math.sin(p * 0.9) };
    for (const key in goal) cam[key] = ease(cam[key], goal[key], dt, rate);
  } else if (camGoal) {
    for (const key in camGoal) cam[key] = ease(cam[key], camGoal[key], dt, 2.5);
  }
  if (stepLens(dt)) for (const r of ribbonSets) r.follow();
  // the cone fades in and its wavefront walks out a few hops a second; solo fades everything but connections
  uniforms.uFocus.value = ease(uniforms.uFocus.value, focusTarget, dt, 4);
  uniforms.uSolo.value = ease(uniforms.uSolo.value, soloTarget, dt, 3);
  uniforms.uSearch.value = ease(uniforms.uSearch.value, searchTarget, dt, 3);
  if (focusTarget) uniforms.uReveal.value = Math.min(MAX_HOPS + 2, uniforms.uReveal.value + dt * 7);
  else if (uniforms.uFocus.value < 0.01) coneLines.visible = false;
  uniforms.uFog.value = selected >= 0 ? Math.max(160, focusDistance * 3) : 160;
  applyCamera();
  camera.updateMatrixWorld();
  placeOverlays(dt);
  uniforms.uTime.value = now / 1000;
  uniforms.uPx.value = renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
  if (selected >= 0) when.textContent = new Date(t0 + daysOf(selected) * 86400000).toISOString().slice(0, 16).replace("T", " ") +
    (zoomed ? "  · zoomed into its moment · F view · Z back out" : "  · Z zoom into its moment") +
    " · ← → connections · Enter go · Backspace back · [ ] dim · I details · Esc sky";
  else {
    const l = overviewLens(), z = Math.min(0, cam.z - 20);
    const d = Math.max(0, Math.min(meta.days, l.anchor - l.tau * Math.expm1(-z / l.scale)));
    when.textContent = new Date(t0 + d * 86400000).toISOString().slice(0, 10) + (autopilot ? "  · autopilot" : "");
  }
  renderer.render(scene, camera);
});
