import test from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../sky.js", import.meta.url), "utf8");
const cardSource = source.slice(source.indexOf("// the focus card"), source.indexOf("// labels on the connections"));

test("unloaded label textures can be disposed during fast navigation", () => {
  let disposed = false;
  const context = vm.createContext({ scene: { remove() {} } });
  vm.runInContext(cardSource, context);
  context.label = { mesh: { material: { dispose() { disposed = true; } } } };
  vm.runInContext("dropCard(label)", context);
  assert.equal(disposed, true);
});

test("a late card request cannot reopen after leaving and returning to the same turn", async () => {
  let resolve;
  const pendingText = new Promise(r => { resolve = r; });
  const context = vm.createContext({ selected: 1, textOf: () => pendingText });
  vm.runInContext(cardSource, context);
  const pending = vm.runInContext("openCard(1)", context);
  vm.runInContext("++cardRun; selected = 1", context);
  resolve({ text: "old request" });
  await pending; // Without the generation check this would attempt to create a stale card.
  assert.equal(vm.runInContext("card", context), null);
});

test("returning to the sky preserves the search constellation", () => {
  const hidden = [];
  const constellation = { show() { hidden.push("search"); } };
  const context = vm.createContext({
    selected: 1, cardRun: 0, zoomed: true, trail: [0], soloTarget: 1, focusTarget: 1,
    setLens() {}, overviewLens() {}, overviewPose: null, camGoal: null,
    ribbonSets: [constellation, { show() { hidden.push("context"); } }], constellation,
    mark: new Float32Array(2), markAttr: {}, clearLabels() {}, closeCard() {}, panel: { style: {} },
  });
  const leaveSource = source.slice(source.indexOf("function leave()"), source.indexOf("// Framing."));
  vm.runInContext(leaveSource + "\nleave();", context);
  assert.deepEqual(hidden, ["context"]);
  assert.equal(context.selected, -1);
});
