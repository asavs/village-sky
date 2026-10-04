import * as THREE from "three";

// World: x, y = the semantic map (similar messages sit together), z = time. The newest message is at z = 0
// and older ones recede on a log scale, so the last weeks spread out up close and the first months become
// the far haze. Stars are additive points sized in world units, so distance does the dimming.
const SPREAD = 60, DEPTH = 400;
let tau = 10;                                   // days; larger keeps the recent past closer to linear

const [meta, starBuf, colorBuf, edgeBuf] = await Promise.all([
  fetch("data/meta.json").then(r => r.json()),
  fetch("data/stars.bin").then(r => r.arrayBuffer()),
  fetch("data/colors.bin").then(r => r.arrayBuffer()),
  fetch("data/edges.bin").then(r => r.arrayBuffer()),
]);
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

const uniforms = { uTime: { value: 0 }, uPx: { value: 1 }, uSize: { value: 0.45 }, uFog: { value: 160 } };
const starMaterial = new THREE.ShaderMaterial({
  uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: /* glsl */`
    attribute vec3 color; attribute float reads;
    uniform float uTime, uPx, uSize, uFog;
    varying vec3 vColor; varying float vBright;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      float h = fract(sin(dot(position.xy, vec2(12.9898, 78.233))) * 43758.5453);
      float twinkle = 0.8 + 0.2 * sin(uTime * (0.8 + 2.0 * h) + h * 40.0);
      float px = uSize * (0.7 + reads / 10.0) * uPx / max(-mv.z, 0.1);
      float fog = 1.0 / (1.0 + pow(-mv.z / uFog, 2.0));   // the far past is a haze, not a pile-up
      vBright = twinkle * fog * min(1.0, (px * px) / 9.0);   // below 3 px a star fades into dust instead of shrinking
      gl_PointSize = clamp(px, 3.0, 96.0);
      vColor = color;
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

// camera: position plus yaw and pitch; wheel flies along the view, left drag looks, right drag pans
const cam = { x: 0, y: 0, z: 80, yaw: 0, pitch: 0 };
const params = new URLSearchParams(location.search);
let autopilot = !params.has("cam"), lastInput = params.has("cam") ? Infinity : -1e9, pilotTime = 0;
if (params.has("cam")) [cam.x, cam.y, cam.z, cam.yaw, cam.pitch] = params.get("cam").split(",").map(Number);
if (params.has("lines")) lines.visible = true;
function forward() {
  return new THREE.Vector3(-Math.sin(cam.yaw) * Math.cos(cam.pitch), Math.sin(cam.pitch), -Math.cos(cam.yaw) * Math.cos(cam.pitch));
}
function applyCamera() {
  camera.position.set(cam.x, cam.y, cam.z);
  camera.lookAt(camera.position.clone().add(forward()));
}
function touched() { autopilot = false; lastInput = performance.now(); }

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
canvas.addEventListener("pointerup", e => { if (drag && drag.moved < 4 && drag.button === 0) pick(e.clientX, e.clientY); drag = null; });
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
  else if (k === "escape") { panel.style.display = "none"; ring.visible = false; }
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
  const s = Math.floor(i / meta.shard);
  if (!shards.has(s)) shards.set(s, fetch(`data/text/${String(s).padStart(4, "0")}.json`).then(r => r.json()));
  const t = (await shards.get(s))[i % meta.shard];
  const c = meta.speakers[rgbs[i * 4 + 3]].color.map(v => Math.round(v * 255)).join(",");
  panel.innerHTML = "";
  const who = Object.assign(document.createElement("div"), { className: "who", textContent: t.speaker });
  who.style.color = `rgb(${c})`;
  const metaLine = Object.assign(document.createElement("div"), { className: "meta",
    textContent: `${t.time} UTC · #${t.room} · read by ${reads[i]} later turns` });
  panel.append(who, metaLine, Object.assign(document.createElement("div"), { className: "text", textContent: t.text }));
  panel.style.display = "block";
  panel.scrollTop = 0;
}

addEventListener("resize", () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
});

const when = document.getElementById("when");
const t0 = Date.parse(meta.first.replace(" ", "T") + "Z");
let previous = performance.now();
renderer.setAnimationLoop(now => {
  const dt = Math.min(0.1, (now - previous) / 1000);
  previous = now;
  if (!autopilot && now - lastInput > 45000 && lastInput !== Infinity) { autopilot = true; pilotTime = 0; }
  if (autopilot) {
    // drift back through time and return, slowly, wandering a little across the map
    pilotTime += dt;
    const p = pilotTime * 0.012;
    cam.z = 80 - DEPTH * 0.55 * (1 - Math.cos(p));
    cam.x = 25 * Math.sin(p * 1.7); cam.y = 15 * Math.sin(p * 1.1);
    cam.yaw = 0.12 * Math.sin(p * 1.3); cam.pitch = 0.08 * Math.sin(p * 0.9);
  }
  applyCamera();
  uniforms.uTime.value = now / 1000;
  uniforms.uPx.value = renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
  const d = Math.max(0, Math.min(meta.days, daysAt(Math.min(0, cam.z - 20))));
  when.textContent = new Date(t0 + d * 86400000).toISOString().slice(0, 10) + (autopilot ? "  · autopilot" : "");
  renderer.render(scene, camera);
});
