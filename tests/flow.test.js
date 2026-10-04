import test from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { readFileSync } from "node:fs";
import * as THREE from "../vendor/three.module.min.js";

const source = readFileSync(new URL("../sky.js", import.meta.url), "utf8");

test("only directed ribbons opt into flow, and reduced-motion changes disable it", () => {
  let change;
  const preference = { matches: false, addEventListener(name, handler) { change = handler; } };
  const time = { value: 10 };
  const ctx = vm.createContext({
    THREE, innerWidth: 800, innerHeight: 600, uniforms: { uTime: time },
    window: { matchMedia: () => preference }, scene: new THREE.Scene(), ECHO_K: 24,
    position: new Float32Array([0, 0, 0, 1, 1, 1]),
  });
  vm.runInContext(source.slice(source.indexOf("const flowPreference"), source.indexOf("// How strongly")), ctx);
  const evaluate = code => vm.runInContext(code, ctx);
  evaluate("contextRibbons.set(0, 0, 1, [1, 1, 1], 2, 1); contextRibbons.show(1)");
  assert.equal(evaluate("contextRibbons.mesh.geometry.attributes.aFlow.array[0]"), 1);
  evaluate("echoRibbons.set(0, 0, 1, [1, 1, 1], 2); echoRibbons.show(1)");
  assert.equal(evaluate("echoRibbons.flow[0]"), 0);
  evaluate("contextRibbons.set(0, 0, 1, [1, 1, 1], 2)");
  assert.equal(evaluate("contextRibbons.flow[0]"), 0); // reused slots don't inherit stale flow
  assert.equal(evaluate("ribbonUniforms.uTime"), time); // uses the existing clock, with no extra frame work
  assert.equal(evaluate("ribbonUniforms.uFlow.value"), 1);
  change({ matches: true });
  assert.equal(evaluate("ribbonUniforms.uFlow.value"), 0);
  change({ matches: false });
  assert.equal(evaluate("ribbonUniforms.uFlow.value"), 1);
});
