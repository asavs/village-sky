// Camera, lens, and card motion use the same damping, including when retargeted mid-flight.
export const MOTION_RATE = 6;
export const damp = (value, target, dt, rate = MOTION_RATE) =>
  value + (target - value) * (1 - Math.exp(-dt * rate));

export function dampAngle(value, target, dt, rate = MOTION_RATE) {
  const delta = Math.atan2(Math.sin(target - value), Math.cos(target - value));
  return damp(value, value + delta, dt, rate);
}

export function sameLens(a, b) {
  return ["anchor", "tau", "scale"].every(key => a[key] === b[key]) &&
    ["centre", "squash"].every(key => a[key].every((v, i) => v === b[key][i]));
}

export const browseLens = (lens, anchor, centre) => ({ ...lens, anchor, centre });
