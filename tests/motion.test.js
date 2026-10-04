import test from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { readFileSync } from "node:fs";
import * as THREE from "../vendor/three.module.min.js";
import { damp, dampAngle, sameLens, browseLens } from "../motion.js";

const source = readFileSync(new URL("../sky.js", import.meta.url), "utf8");
const near = (a, b, epsilon = 1e-4) => assert.ok(Math.abs(a - b) < epsilon, `${a} != ${b}`);

test("camera yaw crosses the angle seam using the short route", () => {
  const from = 179 * Math.PI / 180, to = -179 * Math.PI / 180;
  const halfway = dampAngle(from, to, Math.log(2) / 6);
  near(halfway, Math.PI);
  near(dampAngle(to, from, Math.log(2) / 6), -Math.PI);
});

test("damping follows the same path at different frame rates", () => {
  const integrate = fps => {
    let value = 0;
    for (let frame = 0; frame < fps; frame++) value = damp(value, 100, 1 / fps);
    return value;
  };
  near(integrate(30), integrate(144));
});

test("lens and camera stay coordinated, including a mid-flight reversal", () => {
  const ctx = vm.createContext({
    n: 2, meta: { days: 1 }, tau: 10, SPREAD: 60, DEPTH: 400,
    stars: new Float32Array([0, 0, 0, 0, 1, 1, 1, 0]), rgbs: new Uint8Array(8),
    positionAttr: {}, ribbonSets: [], ease: damp, sameLens,
  });
  ctx.daysOf = i => ctx.stars[i * 4 + 2];
  const lensSource = source.slice(source.indexOf("const overviewLens"), source.indexOf("const renderer"));
  vm.runInContext(lensSource, ctx);
  const evaluate = code => vm.runInContext(code, ctx);
  let cameraX = evaluate("position[0]") + 10;
  evaluate("setLens({...lens, centre: [20, 0], squash: [0.2, 0.4]})");
  let goal = evaluate("toX[0]") + 10;
  for (let frame = 0; frame < 90; frame++) {
    if (frame === 10) {
      const before = evaluate("position[0]");
      evaluate("setLens({...lens, centre: [-10, 0]})");
      near(evaluate("position[0]"), before); // retargeting does not snap positions
      goal = evaluate("toX[0]") + 10;
    }
    evaluate("stepLens(1 / 60)");
    cameraX = damp(cameraX, goal, 1 / 60);
    near(cameraX - evaluate("position[0]"), 10);
  }
  const progress = evaluate("lensBlend");
  evaluate("setLens({...lens, centre: [...lens.centre], squash: [...lens.squash]})");
  assert.equal(evaluate("lensBlend"), progress); // an unchanged lens doesn't restart animation
});

test("zoomed arrow steps retain scale, framing, and the pending camera offset", () => {
  const contextThreads = [], dayThreads = [];
  const ctx = vm.createContext({
    THREE, SPREAD: 60, n: 3, stars: new Float32Array([0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0]),
    lens: { anchor: 0, tau: 0.1, scale: 30, squash: [0.2, 0.4], centre: [0, 0] },
    selected: 0, zoomed: true, focusDistance: 32, cam: { x: -25, y: 3, z: 6, yaw: 1, pitch: 0.1 },
    camGoal: { x: -32, y: 3, z: 6, yaw: 1, pitch: 0.1 },
    overviewPose: null, trail: [], autopilot: false, lastInput: 0, facing: 1, focusCtx: null,
    params: new URLSearchParams(), performance, browseLens, setLens(next) { ctx.lens = next; },
    viewPose() { return { x: 0, y: 0, z: 20, yaw: 0, pitch: 0 }; },
    frameFor() { throw new Error("Browsing must not reframe"); },
    contextOf: i => ({ inputs: i > 0 ? [i - 1] : [], readers: i < 2 ? [i + 1] : [] }), lightCone: () => ({ past: [], future: [] }), echoesOf: () => [],
    mark: new Float32Array(3), markAttr: {}, warpPrev: [-1, 0, 1], warpNext: [1, 2, -1],
    soloTarget: 0, WARP_SHOWN: 16, ECHO_LABELS: 6,
    contextRibbons: { set(...args) { contextThreads.push(args); }, show() {} },
    warpRibbons: { set(...args) { dayThreads.push(args); }, show() {} }, worldline: () => [1, 1, 1],
    buildLabels() {}, openCard() {}, fillPanel() {},
    overviewLens: () => ({ anchor: 2, tau: 10, scale: 100, squash: [1, 1], centre: [0, 0] }),
  });
  vm.runInContext(source.slice(source.indexOf("const xIn"), source.indexOf("let lens =")), ctx);
  ctx.daysOf = i => ctx.stars[i * 4 + 2];
  vm.runInContext(source.slice(source.indexOf("function zoomInto("), source.indexOf("function toggleZoom(")), ctx);
  vm.runInContext(source.slice(source.indexOf("const pointInLens"), source.indexOf("function worldline(")), ctx);
  vm.runInContext("focus(1, true, true)", ctx);
  assert.deepEqual(contextThreads.map(t => t.slice(1, 3)), [[0, 1], [1, 2], [0, 1], [1, 2]]);
  assert.deepEqual(contextThreads.map(t => t[5]), [0.55, 0.55, 1, 1]);
  assert.deepEqual(dayThreads.map(t => t.slice(1, 3)), [[0, 1], [1, 2]]);
  assert.ok(dayThreads.every(t => t[5] === 0.55));
  near(ctx.camGoal.x, 60 - 32);
  ctx.cam.x = damp(ctx.cam.x, ctx.camGoal.x, 0.05);
  vm.runInContext("focus(2, true, true); focus(1, true, true)", ctx);
  near(ctx.camGoal.x, 60 - 32);
  near(ctx.camGoal.y, 3); near(ctx.camGoal.z, 6);
  near(ctx.camGoal.yaw, 1); near(ctx.camGoal.pitch, 0.1);
  near(ctx.lens.tau, 0.1); near(ctx.lens.scale, 30); near(ctx.focusDistance, 32);
});

function cardHarness(textOf) {
  const textures = [];
  const ctx = vm.createContext({
    THREE, selected: 0, textOf, memeOf: null, params: new URLSearchParams(), scene: new THREE.Scene(),
    ribbonUniforms: { uResolution: { value: new THREE.Vector2(800, 600) } }, uniforms: { uFog: { value: 160 } },
    speakerColor: i => i ? [0, 1, 0] : [1, 0, 0], rgbOf: () => "white", position: new Float32Array(12),
    renderer: { domElement: { clientHeight: 600 } }, camera: { fov: 60 }, focusDistance: 32, innerWidth: 800, innerHeight: 600,
    textFailed() { throw new Error("Unexpected loading error"); },
    textCanvas() {
      const texture = { disposed: false, dispose() { this.disposed = true; } };
      textures.push(texture);
      return { texture, w: 460, h: 200 };
    },
  });
  vm.runInContext(source.slice(source.indexOf("const cardUniforms"), source.indexOf("// draw text")), ctx);
  vm.runInContext(source.slice(source.indexOf("// the focus card"), source.indexOf("// labels on the connections")), ctx);
  return { ctx, textures, evaluate: code => vm.runInContext(code, ctx) };
}

test("arrow navigation retains the card mesh and reveal state while replacing text", async () => {
  const h = cardHarness(async i => ({ speaker: `agent ${i}`, time: "2026-01-01", text: `turn ${i}` }));
  await h.evaluate("openCard(0)");
  const mesh = h.evaluate("card.mesh");
  h.evaluate("card.open = 1; selected = 1");
  await h.evaluate("openCard(1, true)");
  assert.equal(h.evaluate("card.mesh"), mesh);
  assert.equal(h.evaluate("card.open"), 1);
  assert.equal(h.evaluate("card.j"), 1);
  assert.equal(h.textures[0].disposed, true);
  assert.equal(h.textures[1].disposed, false);
  assert.equal(h.ctx.scene.children.length, 1);
});

test("out-of-order text cannot overwrite a retained card during rapid stepping", async () => {
  const resolvers = new Map();
  const h = cardHarness(i => i === 0 ? Promise.resolve({ speaker: "agent", time: "2026", text: "first" })
    : new Promise(resolve => resolvers.set(i, resolve)));
  await h.evaluate("openCard(0)");
  h.evaluate("card.open = 1; selected = 1");
  const first = h.evaluate("openCard(1, true)");
  assert.equal(h.evaluate("card.mesh.material.uniforms.uTextAlpha.value"), 0);
  h.evaluate("selected = 2");
  const second = h.evaluate("openCard(2, true)");
  resolvers.get(2)({ speaker: "agent", time: "2026", text: "newest" });
  await second;
  const texture = h.evaluate("card.texture");
  resolvers.get(1)({ speaker: "agent", time: "2026", text: "late" });
  await first;
  assert.equal(h.evaluate("card.j"), 2);
  assert.equal(h.evaluate("card.texture"), texture);
  assert.equal(h.evaluate("card.mesh.material.uniforms.uTextAlpha.value"), 1);
  assert.equal(h.textures.length, 2);
});

test("shared labels survive steps, departed labels retire, and reappearing labels are reused", async () => {
  const h = cardHarness(async i => ({ speaker: `agent ${i}`, text: `turn ${i}`, time: "2026" }));
  Object.assign(h.ctx, { labels: [], retiringLabels: [], cursor: -1, ago: () => "1 min", daysOf: i => i });
  h.ctx.textCanvas = () => ({ texture: { dispose() {} }, w: 100, h: 30 });
  vm.runInContext(source.slice(source.indexOf("// labels on the connections"), source.indexOf("function cycle(")), h.ctx);
  h.evaluate("buildLabels(0, [{j: 1, tag: 'read'}, {j: 2, tag: 'echo'}])");
  await Promise.resolve(); await Promise.resolve();
  const shared = h.ctx.labels[0], departed = h.ctx.labels[1];
  shared.alpha = 0.8; departed.alpha = 0.6; h.ctx.cursor = 0;
  h.evaluate("buildLabels(1, [{j: 1, tag: 'previous'}], true)");
  assert.equal(h.ctx.labels[0], shared);
  near(shared.alpha, 0.8);
  assert.equal(h.ctx.cursor, 0);
  assert.equal(h.ctx.retiringLabels[0], departed);
  h.evaluate("buildLabels(2, [{j: 1, tag: 'read'}, {j: 2, tag: 'echo'}], true)");
  assert.equal(h.ctx.labels[1], departed);
  near(departed.alpha, 0.6);
  assert.equal(h.ctx.retiringLabels.length, 0);
  h.evaluate("clearLabels()");
  assert.equal(h.ctx.scene.children.length, 0);
});

test("a retained card moves with the camera without jumping to the new star", async () => {
  const h = cardHarness(async i => ({ speaker: `agent ${i}`, time: "2026", text: `turn ${i}` }));
  Object.assign(h.ctx, { labels: [], retiringLabels: [], cursor: -1, MOTION_RATE: 6, ease: damp,
    screenOf: () => null, pointInLens: () => new THREE.Vector3(60, 0, 0) });
  await h.evaluate("openCard(0)");
  h.evaluate("card.open = 1; selected = 1");
  await h.evaluate("openCard(1, true)");
  near(h.evaluate("card.mesh.material.uniforms.uCenter.value.x"), 0);
  vm.runInContext(source.slice(source.indexOf("function placeOverlays("), source.indexOf("function textFailed(")), h.ctx);
  let cameraX = -32;
  for (let frame = 0; frame < 30; frame++) {
    h.evaluate("placeOverlays(1 / 60)");
    cameraX = damp(cameraX, 60 - 32, 1 / 60);
    near(cameraX - h.evaluate("card.mesh.material.uniforms.uCenter.value.x"), -32);
  }
});

test("retired labels fade and release their mesh, material, and texture", async () => {
  const h = cardHarness(async i => ({ speaker: `agent ${i}`, text: "turn", time: "2026" }));
  Object.assign(h.ctx, { labels: [], retiringLabels: [], cursor: -1, ago: () => "1 min", daysOf: i => i,
    screenOf: () => null });
  vm.runInContext(source.slice(source.indexOf("// labels on the connections"), source.indexOf("function cycle(")), h.ctx);
  vm.runInContext(source.slice(source.indexOf("function placeOverlays("), source.indexOf("function textFailed(")), h.ctx);
  h.evaluate("buildLabels(0, [{j: 1, tag: 'read'}])");
  await Promise.resolve(); await Promise.resolve();
  const l = h.ctx.labels[0];
  let materialDisposed = false;
  l.mesh.material.addEventListener("dispose", () => { materialDisposed = true; });
  l.alpha = 0.6; l.mesh.visible = true;
  h.evaluate("buildLabels(1, [], true); placeOverlays(0.05)");
  near(l.alpha, 0.3);
  assert.equal(h.ctx.scene.children.length, 1);
  h.evaluate("placeOverlays(0.05)");
  assert.equal(h.ctx.retiringLabels.length, 0);
  assert.equal(h.ctx.scene.children.length, 0);
  assert.equal(materialDisposed, true);
  assert.equal(h.textures[0].disposed, true);
});
