import * as THREE from "three";

// World: x, y = the semantic map (similar messages sit together), z = time. The newest message is at z = 0
// and older ones recede on a log scale, so the last weeks spread out up close and the first months become
// the far haze. Stars are additive points sized in world units, so distance does the dimming.
const SPREAD = 60, DEPTH = 400;
let tau = 10;                                   // days; larger keeps the recent past closer to linear

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
document.getElementById("counts").textContent =
  `${n.toLocaleString()} turns · ${meta.speakers.length} speakers · ${meta.first.slice(0, 10)} to ${meta.last.slice(0, 10)}`;

const depthOf = days => -DEPTH * Math.log1p((meta.days - days) / tau) / Math.log1p(meta.days / tau);
const daysAt = z => meta.days - tau * Math.expm1(-z / DEPTH * Math.log1p(meta.days / tau));

const position = new Float32Array(n * 3), color = new Float32Array(n * 3), reads = new Float32Array(n);
function place() {
  for (let i = 0; i < n; i++) {
    position[i * 3] = stars[i * 4] * SPREAD;
    position[i * 3 + 1] = stars[i * 4 + 1] * SPREAD;
    position[i * 3 + 2] = depthOf(stars[i * 4 + 2]);
  }
}
place();
for (let i = 0; i < n; i++) {
  for (let c = 0; c < 3; c++) color[i * 3 + c] = rgbs[i * 4 + c] / 255;
  reads[i] = stars[i * 4 + 3];
}

const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 4000);

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
    const want = targets.filter(j => (stars[j * 4 + 2] - stars[i * 4 + 2]) * side >= 0 && j !== i);
    if (!want.length) continue;
    const bound = side < 0 ? Math.min(...want.map(j => stars[j * 4 + 2])) : Math.max(...want.map(j => stars[j * 4 + 2]));
    const wanted = new Set(want);
    let head = 0, tail = 0;
    handoffs[i] = 0; touchedNodes.push(i); deque[tail++] = i;
    while (head !== tail && wanted.size) {
      const a = deque[head]; head = (head + 1) % size;
      if (wanted.delete(a)) result.set(a, side * handoffs[a]);
      for (let k = adj.start[a]; k < adj.start[a + 1]; k++) {
        const b = adj.list[k];
        if ((stars[b * 4 + 2] - bound) * side > 0) continue;
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

// cone per star: 0 the selected star, -h h hops into its past, +h h hops into its future, NONE outside
const NONE = 1e4, MAX_HOPS = 40;
const cone = new Float32Array(n).fill(NONE);
const coneAttr = new THREE.BufferAttribute(cone, 1);
geometry.setAttribute("cone", coneAttr);

const uniforms = { uTime: { value: 0 }, uPx: { value: 1 }, uSize: { value: 0.45 }, uFog: { value: 160 },
  uFocus: { value: 0 }, uReveal: { value: 0 } };
const starMaterial = new THREE.ShaderMaterial({
  uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: /* glsl */`
    attribute vec3 color; attribute float reads, cone;
    uniform float uTime, uPx, uSize, uFog, uFocus, uReveal;
    varying vec3 vColor; varying float vBright;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      float h = fract(sin(dot(position.xy, vec2(12.9898, 78.233))) * 43758.5453);
      float twinkle = 0.8 + 0.2 * sin(uTime * (0.8 + 2.0 * h) + h * 40.0);
      float hops = abs(cone);
      float lit = step(hops, uReveal) * step(cone, 9000.0);                  // inside the cone and reached already
      float front = lit * exp(-pow((uReveal - hops) * 1.2, 2.0));           // the wavefront flares as it passes
      float grow = mix(1.0, mix(1.0, 1.0 + exp(-hops / 6.0), lit), uFocus);
      float px = grow * uSize * (0.7 + reads / 10.0) * uPx / max(-mv.z, 0.1);
      float fog = 1.0 / (1.0 + pow(-mv.z / uFog, 2.0));   // the far past is a haze, not a pile-up
      fog = mix(fog, sqrt(fog), lit * uFocus);              // the cone carries further through the haze
      float weight = lit * (0.12 + 1.6 * exp(-hops / 5.0)) + front * 1.5;       // near hops blaze, the far cone glows
      vBright = mix(1.0, mix(0.07, weight, lit), uFocus) * twinkle * fog * min(1.0, (px * px) / 9.0);
      gl_PointSize = clamp(px, 3.0, 96.0);
      vec3 tint = cone < 0.0 ? vec3(1.0, 0.78, 0.45) : vec3(0.55, 0.8, 1.0);   // past warm, future cool
      vColor = mix(color, tint, 0.5 * lit * uFocus * step(0.5, hops));
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

// the cone's own edges: each one lights when the wavefront reaches its far end
const coneLineGeometry = new THREE.BufferGeometry();
coneLineGeometry.setAttribute("position", positionAttr);
coneLineGeometry.setAttribute("color", geometry.getAttribute("color"));
coneLineGeometry.setAttribute("cone", coneAttr);
coneLineGeometry.setIndex(new THREE.BufferAttribute(new Uint32Array(2), 1));
const coneLines = new THREE.LineSegments(coneLineGeometry, new THREE.ShaderMaterial({
  uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: `attribute vec3 color; attribute float cone; uniform float uReveal, uFocus;
    varying vec3 vColor; varying float vAlpha;
    void main() {
      float hops = abs(cone);
      vAlpha = uFocus * step(hops, uReveal) * (0.006 + 0.35 * exp(-hops / 3.0));
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
  if (params.has("reveal")) { uniforms.uFocus.value = 1; uniforms.uReveal.value = MAX_HOPS + 2; }   // captures skip the animation
  return { past, future };
}
function clearCone() { focusTarget = 0; echoLines.visible = echoRings.visible = false; }
let focusTarget = 0;

// the selected star: a ring around it
const ringGeometry = new THREE.BufferGeometry();
ringGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
const ring = new THREE.Points(ringGeometry, new THREE.ShaderMaterial({
  transparent: true, depthTest: false, blending: THREE.AdditiveBlending,
  vertexShader: `void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_PointSize = 44.0; }`,
  fragmentShader: `void main() { float r = length(gl_PointCoord * 2.0 - 1.0);
    gl_FragColor = vec4(1.0, 0.95, 0.85, exp(-pow((r - 0.8) * 14.0, 2.0))); }`,
}));
ring.visible = false;
scene.add(ring);

// echoes: the selected star's nearest neighbours in meaning, joined to it by a thread coloured by contact.
// gold = within 2 hops of context (likely carried), white = 3 to 8 hops, violet = no contact within 8
// (likely found independently), grey = the same speaker again
const ECHO_K = 24, ECHO_MIN = 0.8;
const echoIndex = echoIndexBuf && new Uint32Array(echoIndexBuf), echoSim = echoSimBuf && new Uint8Array(echoSimBuf);
const CONTACT = { close: [1.0, 0.8, 0.35], near: [0.85, 0.85, 0.9], far: [0.45, 0.6, 0.85], none: [0.8, 0.45, 1.0],
  self: [0.4, 0.4, 0.45] };
const GROUPS = [["close", "in contact", "1 hand-off: one read the other, then or from memory"],
  ["near", "near", "2 to 3 hand-offs: through intermediaries"], ["far", "distant", "4 or more hand-offs"],
  ["none", "no contact", "no path through context: found independently"],
  ["self", "itself", "the same speaker, from its own memory"]];
// each echo is a ribbon from the selected star: a quad expanded in screen space, so its width is in pixels.
// Width and brightness carry the strength of the connection (see strengthOf).
const echoLineGeometry = new THREE.InstancedBufferGeometry();
echoLineGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array([0, -1, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0]), 3));
echoLineGeometry.setIndex([0, 1, 2, 0, 2, 3]);
const ribbon = { start: new Float32Array(ECHO_K * 3), end: new Float32Array(ECHO_K * 3), color: new Float32Array(ECHO_K * 3),
  width: new Float32Array(ECHO_K) };
echoLineGeometry.setAttribute("aStart", new THREE.InstancedBufferAttribute(ribbon.start, 3));
echoLineGeometry.setAttribute("aEnd", new THREE.InstancedBufferAttribute(ribbon.end, 3));
echoLineGeometry.setAttribute("aColor", new THREE.InstancedBufferAttribute(ribbon.color, 3));
echoLineGeometry.setAttribute("aWidth", new THREE.InstancedBufferAttribute(ribbon.width, 1));
const ribbonUniforms = { uResolution: { value: new THREE.Vector2(innerWidth, innerHeight) } };
const echoLines = new THREE.Mesh(echoLineGeometry, new THREE.ShaderMaterial({
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
      gl_FragColor = vec4(vColor, edge * mix(0.35, 1.0, vAlong));           // dimmer at the selected star
    }`,
}));
echoLines.frustumCulled = false;
const echoRingPos = new Float32Array(ECHO_K * 3), echoRingCol = new Float32Array(ECHO_K * 3);
const echoRingGeometry = new THREE.BufferGeometry();
echoRingGeometry.setAttribute("position", new THREE.BufferAttribute(echoRingPos, 3));
echoRingGeometry.setAttribute("color", new THREE.BufferAttribute(echoRingCol, 3));
const echoRings = new THREE.Points(echoRingGeometry, new THREE.ShaderMaterial({
  transparent: true, depthTest: false, blending: THREE.AdditiveBlending,
  vertexShader: `attribute vec3 color; varying vec3 vColor;
    void main() { vColor = color; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_PointSize = 26.0; }`,
  fragmentShader: `varying vec3 vColor; void main() { float r = length(gl_PointCoord * 2.0 - 1.0);
    gl_FragColor = vec4(vColor, exp(-pow((r - 0.75) * 10.0, 2.0))); }`,
}));
echoLines.visible = echoRings.visible = false;
scene.add(echoLines, echoRings);

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
function showEchoes(i) {
  if (!echoIndex) return [];
  const list = [];
  for (let k = 0; k < ECHO_K; k++) {
    const j = echoIndex[i * ECHO_K + k], sim = echoSim[i * ECHO_K + k] / 255;
    if (sim < ECHO_MIN) break;
    list.push({ j, sim });
  }
  const other = list.filter(e => rgbs[e.j * 4 + 3] !== rgbs[i * 4 + 3]).map(e => e.j);
  const handoffs = contactDistances(i, other, 0), hops = contactDistances(i, other, 1);
  for (const e of list) {
    e.handoffs = handoffs.get(e.j); e.hops = hops.get(e.j);
    e.contact = contactOf(i, e.j, e.handoffs);
    e.strength = strengthOf(e);
  }
  list.forEach((e, k) => {
    const bright = 0.25 + 0.95 * e.strength;
    ribbon.start.set(position.subarray(i * 3, i * 3 + 3), k * 3);
    ribbon.end.set(position.subarray(e.j * 3, e.j * 3 + 3), k * 3);
    ribbon.color.set(CONTACT[e.contact].map(v => v * bright), k * 3);
    ribbon.width[k] = 1 + 9 * e.strength;
    echoRingPos.set(position.subarray(e.j * 3, e.j * 3 + 3), k * 3);
    echoRingCol.set(CONTACT[e.contact].map(v => v * bright), k * 3);
  });
  for (const g of [echoLineGeometry, echoRingGeometry]) for (const a of Object.values(g.attributes)) a.needsUpdate = true;
  echoLineGeometry.instanceCount = list.length;
  echoRingGeometry.setDrawRange(0, list.length);
  echoLines.visible = echoRings.visible = list.length > 0;
  return list.sort((a, b) => stars[a.j * 4 + 2] - stars[b.j * 4 + 2]);
}

// camera: position plus yaw and pitch; wheel flies along the view, left drag looks, right drag pans
const cam = { x: 0, y: 0, z: 80, yaw: 0, pitch: 0 };
const params = new URLSearchParams(location.search);
let autopilot = !params.has("cam"), lastInput = params.has("cam") ? Infinity : -1e9, pilotTime = 0;
if (params.has("cam")) [cam.x, cam.y, cam.z, cam.yaw, cam.pitch] = params.get("cam").split(",").map(Number);
if (params.has("lines")) lines.visible = true;
if (params.has("open")) setTimeout(() => open(Number(params.get("open"))), 0);
function forward() {
  return new THREE.Vector3(-Math.sin(cam.yaw) * Math.cos(cam.pitch), Math.sin(cam.pitch), -Math.cos(cam.yaw) * Math.cos(cam.pitch));
}
function applyCamera() {
  camera.position.set(cam.x, cam.y, cam.z);
  camera.lookAt(camera.position.clone().add(forward()));
}
function touched() { autopilot = false; lastInput = performance.now(); camGoal = null; }
// glide to look at star i from a little nearer the present
let camGoal = null;
function flyTo(i) {
  autopilot = false; lastInput = performance.now();
  camGoal = { x: position[i * 3], y: position[i * 3 + 1], z: position[i * 3 + 2] + 45, yaw: 0, pitch: 0 };
}

const canvas = renderer.domElement;
let drag = null;
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, button: e.button, moved: 0 }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.moved += Math.abs(dx) + Math.abs(dy);
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.moved < 4) return;
  touched();
  if (drag.button === 0) {
    cam.yaw += dx * 0.003;
    cam.pitch = Math.max(-1.4, Math.min(1.4, cam.pitch - dy * 0.003));
  } else {
    const k = 0.0015 * Math.max(20, Math.abs(cam.z) * 0.3 + 40);
    const right = new THREE.Vector3().crossVectors(forward(), new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3().crossVectors(right, forward());
    cam.x += (-dx * right.x + dy * up.x) * k; cam.y += (-dx * right.y + dy * up.y) * k; cam.z += (-dx * right.z + dy * up.z) * k;
  }
});
canvas.addEventListener("pointerup", e => { if (drag && drag.moved < 4 && drag.button === 0) { touched(); pick(e.clientX, e.clientY); } drag = null; });
canvas.addEventListener("wheel", e => {
  e.preventDefault(); touched();
  const f = forward(), step = -e.deltaY * 0.08 * Math.max(1, Math.abs(cam.z) / 60);
  cam.x += f.x * step; cam.y += f.y * step; cam.z += f.z * step;
}, { passive: false });

addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (k === "l") lines.visible = !lines.visible;
  else if (k === "a") { autopilot = !autopilot; pilotTime = 0; }
  else if (k === "h") document.body.classList.toggle("hide-ui");
  else if (k === "escape") { panel.style.display = "none"; ring.visible = false; clearCone(); }
  else if (k === "[" || k === "]") { tau *= k === "]" ? 1.5 : 1 / 1.5; place(); positionAttr.needsUpdate = true; }
  else if (k === "=" || k === "-") uniforms.uSize.value *= k === "=" ? 1.2 : 1 / 1.2;
  else if (k === "." || k === ",") lineUniforms.uAlpha.value *= k === "." ? 1.5 : 1 / 1.5;
});

// picking: project every star, take the nearest to the pointer within 14 px, preferring the closest to the camera
const panel = document.getElementById("panel"), shards = new Map();
const viewProjection = new THREE.Matrix4();
function pick(sx, sy) {
  viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const m = viewProjection.elements, w2 = innerWidth / 2, h2 = innerHeight / 2;
  let best = -1, bestScore = 14 * 14;
  for (let i = 0; i < n; i++) {
    const x = position[i * 3], y = position[i * 3 + 1], z = position[i * 3 + 2];
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (w <= 0.1) continue;
    const px = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / w + 1) * w2 - sx;
    const py = (1 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w) * h2 - sy;
    const score = px * px + py * py + w * 0.02;
    if (score < bestScore) { bestScore = score; best = i; }
  }
  if (best >= 0) open(best);
}
async function open(i) {
  ringGeometry.attributes.position.array.set(position.subarray(i * 3, i * 3 + 3));
  ringGeometry.attributes.position.needsUpdate = true;
  ring.visible = true;
  const { past, future } = lightCone(i);
  const echoes = showEchoes(i);
  const s = Math.floor(i / meta.shard);
  if (!shards.has(s)) shards.set(s, fetch(`data/text/${String(s).padStart(4, "0")}.json`).then(r => r.json()));
  const t = (await shards.get(s))[i % meta.shard];
  const c = meta.speakers[rgbs[i * 4 + 3]].color.map(v => Math.round(v * 255)).join(",");
  panel.innerHTML = "";
  const who = Object.assign(document.createElement("div"), { className: "who", textContent: t.speaker });
  who.style.color = `rgb(${c})`;
  const metaLine = Object.assign(document.createElement("div"), { className: "meta",
    textContent: `${t.time} UTC · #${t.room} · read by ${reads[i]} later turns` });
  panel.append(who, metaLine, Object.assign(document.createElement("div"), { className: "text", textContent: t.text }),
    coneSummary("past cone", "could have shaped it", past, "back to", "#ffc773"),
    coneSummary("future cone", "it could have reached", future, "out to", "#8cccff"),
    echoList(i, echoes));
  panel.style.display = "block";
  panel.scrollTop = 0;
}

// the echoes, in time order, each with its contact: how many hops of context lie between the two turns
const textOf = async j => {
  const s = Math.floor(j / meta.shard);
  if (!shards.has(s)) shards.set(s, fetch(`data/text/${String(s).padStart(4, "0")}.json`).then(r => r.json()));
  return (await shards.get(s))[j % meta.shard];
};
const rgbOf = c => `rgb(${c.map(v => Math.round(v * 255)).join(",")})`;
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
    for (const { j, sim, hops, handoffs } of members) {
      const item = Object.assign(document.createElement("div"), { className: "echo" });
      const when = stars[j * 4 + 2] < stars[i * 4 + 2] ? "earlier" : "later";
      const h = Math.abs(handoffs), distance = handoffs === undefined ? "" : (h === 0 ? " · same thread"
        : ` · ${h} hand-off${h > 1 ? "s" : ""}`) + (hops === undefined ? "" : ` · ${Math.abs(hops)} hops fresh`);
      const line = Object.assign(document.createElement("div"), { className: "line" });
      const snip = Object.assign(document.createElement("div"), { className: "snip" });
      item.append(line, snip);
      item.addEventListener("click", () => { flyTo(j); open(j); });
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
    if ((stars[j * 4 + 2] - stars[far * 4 + 2]) * cone[j] > 0) far = j;     // earliest in the past, latest in the future
  }
  const date = j => new Date(t0 + stars[j * 4 + 2] * 86400000).toISOString().slice(0, 16).replace("T", " ");
  const top = [...counts].sort((a, b) => b[1] - a[1]).slice(0, 6)
    .map(([k, c]) => `${meta.speakers[k].name} ${c}`).join(" · ");
  box.innerHTML = "";
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

const when = document.getElementById("when");
let previous = performance.now();
renderer.setAnimationLoop(now => {
  const dt = Math.min(0.1, (now - previous) / 1000);
  previous = now;
  if (!autopilot && now - lastInput > 45000 && lastInput !== Infinity) { autopilot = true; pilotTime = 0; }
  if (autopilot) {
    // drift back through time and return, slowly, wandering a little across the map; glide into the path
    pilotTime += dt;
    const p = pilotTime * 0.012, ease = 1 - Math.exp(-dt * Math.min(2, 0.05 + pilotTime * 0.05));
    const goal = { z: 80 - DEPTH * 0.55 * (1 - Math.cos(p)), x: 25 * Math.sin(p * 1.7), y: 15 * Math.sin(p * 1.1),
      yaw: 0.12 * Math.sin(p * 1.3), pitch: 0.08 * Math.sin(p * 0.9) };
    for (const key in goal) cam[key] += (goal[key] - cam[key]) * ease;
  } else if (camGoal) {
    const ease = 1 - Math.exp(-dt * 2.5);
    for (const key in camGoal) cam[key] += (camGoal[key] - cam[key]) * ease;
  }
  // the cone fades in, and its wavefront walks out a few hops a second
  const f = uniforms.uFocus;
  f.value += (focusTarget - f.value) * (1 - Math.exp(-dt * 4));
  if (focusTarget) uniforms.uReveal.value = Math.min(MAX_HOPS + 2, uniforms.uReveal.value + dt * 7);
  else if (f.value < 0.01) coneLines.visible = false;
  applyCamera();
  uniforms.uTime.value = now / 1000;
  uniforms.uPx.value = renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
  const d = Math.max(0, Math.min(meta.days, daysAt(Math.min(0, cam.z - 20))));
  when.textContent = new Date(t0 + d * 86400000).toISOString().slice(0, 10) + (autopilot ? "  · autopilot" : "");
  renderer.render(scene, camera);
});
