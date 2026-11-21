// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only).
//
// Forecasts each GTA VI PS5 trophy's Exophase earn rate at a case's target
// as five quantiles, from the reads the house recorded before the checkpoint.
//
// The core idea: a rate is earners over players-tracked, so the two are
// modelled separately and the rate is their ratio, never extrapolated as a
// rate alone. Exophase refreshes a game's counts on its own schedule, not
// with every trophy earned, so when the tracked population jumps at a
// refresh the rates dilute even while absolute earners grow. That is what
// the launch-night practice shows: 638 players with frozen rates, then one
// refresh to 3,868 players and nearly every rate lower.
//
// Method, per case:
//   - Distinct snapshots: consecutive reads that are byte-for-byte the same
//     page are one stale snapshot; only the first read showing it counts.
//   - Flat history (one distinct snapshot, e.g. launch night): the page is
//     stale and the next refresh reveals launch growth accumulated since
//     the last refresh, so the population is expected to jump and rates to
//     dilute by nearly the whole jump - the newcomers are brand new and
//     hold almost nothing beyond the first trophy, so the median shifts
//     down by the population-jump factor with only a small offset for the
//     few trophies newcomers earn, and the spread stays wide.
//   - Real history (two or more distinct snapshots): log-log fits of
//     log players and log earners against ln(1 + hours since release) over
//     the recent window, each shrunk toward a prior slope, averaged in
//     logit space with a shrunk log-linear fit of logit(rate) itself.
//     Earners cannot outrun the population for long and rates are capped
//     in how far they may move from the last read.
//   - Spread: a base term for refresh/model error, a term that grows with
//     the square root of the horizon, each trophy's fit residual, and an
//     extra term for rare trophies (a handful of new earners swings their
//     logit a lot). A trophy last read at 100% is forecast to stay there.
//   - Defensive fallback: if the model ever throws for a case, the output
//     is still a valid persistence forecast with a random-walk spread, so
//     a bug can never turn a case invalid.
// Deterministic: no randomness, no network, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** A rate is clamped to this percent range before any logit, as the scorer does. */
const LO = 0.01;
const HI = 99.99;
/** Hours in a day. */
const DAY_H = 24;

const clampPct = (x: number): number => Math.min(HI, Math.max(LO, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

interface Trophy {
  id: number;
  name: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date?: string };
  trophies: Trophy[];
  history: Read[];
}

/** One distinct page state: the first read that showed it, and its values. */
interface Snapshot {
  ms: number; // first read time, epoch ms
  players: number;
  rates: Array<number | null>;
}

/** Least-squares line y = a + b x, with the residual spread. */
function ols(xs: number[], ys: number[]): { a: number; b: number; resid: number } {
  const n = xs.length;
  if (n === 0) return { a: 0, b: 0, resid: 0 };
  if (n === 1) return { a: ys[0], b: 0, resid: 0 };
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    ss += r * r;
  }
  const resid = n > 2 ? Math.sqrt(ss / (n - 2)) : 0;
  return { a, b, resid };
}

/** Shrink a fitted slope toward a prior slope; weight 1 is the prior's strength. */
function shrinkSlope(fit: number, prior: number, n: number): number {
  return (n * fit + 1 * prior) / (n + 1);
}

/** Collapse consecutive identical reads into distinct page snapshots. */
function snapshotsOf(history: Read[]): Snapshot[] {
  const snaps: Snapshot[] = [];
  for (const read of history) {
    const ms = Date.parse(read.at);
    const prev = snaps[snaps.length - 1];
    const same =
      prev !== undefined &&
      prev.players === read.players_tracked &&
      prev.rates.length === read.rates.length &&
      prev.rates.every((r, i) => r === read.rates[i]);
    if (same) continue;
    snaps.push({ ms, players: read.players_tracked, rates: read.rates.slice() });
  }
  return snaps;
}

/** Hours from the release midnight to a moment; negative times count as 0. */
function hoursSinceRelease(ms: number, releaseMs: number): number {
  return Math.max(0, (ms - releaseMs) / 3_600_000);
}

/** x for the log-time fits: ln(1 + hours since release). */
function logTime(ms: number, releaseMs: number): number {
  return Math.log(1 + hoursSinceRelease(ms, releaseMs));
}

/**
 * Flat history: one distinct snapshot. The page is stale, so the next
 * refresh reveals launch growth accumulated since the last refresh: the
 * snapshot was already stale when first read (a typical refresh interval
 * before it) plus the stale time to the checkpoint plus the horizon.
 * The population grows fast during launch (the rules' refresh episode
 * implies ~30% an hour in log terms), and the jump is capped at the
 * largest documented one, 6x. Crucially, the newcomers are brand new:
 * in that episode the tracked population grew 6.06x while the trophies
 * earned grew only 4.54x, and the one trophy at 100% accounts for 638 of
 * the 943 launch-night earners, so the other 54 trophies' earners grew
 * only 1.34x against the 6.06x population - the new players held almost
 * nothing beyond the first trophy (an aggregate earn fraction near 0.07).
 * So the rate dilutes by nearly the whole population jump: the median is
 * the last rate times (1 + f * (G - 1)) / G, with f the small fraction
 * of newcomers who earn each trophy, mildly higher for the easy trophies
 * everyone gets first. The spread stays wide because the jump's size and
 * the per-trophy split of the newcomers' few trophies are both uncertain.
 */
function flatForecast(
  snaps: Snapshot[],
  trophies: Trophy[],
  checkpointMs: number,
  horizonMinutes: number,
): Array<{ id: number; q: number[] }> {
  const snap = snaps[0];
  const staleHours = Math.max(0, (checkpointMs - snap.ms) / 3_600_000);
  const horizonHours = horizonMinutes / 60;
  const growthTime = 2 + staleHours + horizonHours;
  const popRate = 0.3; // log players per hour during the launch surge
  const jump = Math.min(6, Math.exp(popRate * growthTime));
  const horizonDays = horizonMinutes / 1440;
  return trophies.map((t, i) => {
    const last = snap.rates[i];
    const lastRate = typeof last === 'number' ? last : 1;
    if (lastRate >= HI) {
      // A trophy every tracked player holds stays (near) 100%: it survived
      // every refresh so far, and on the logit scale even 99% is far from
      // 100%, so the lower tail stays tight.
      return { id: t.id, q: [99.9, 99.98, 99.99, 99.99, 99.99] };
    }
    // Fraction of the newcomers who earn this trophy in their first hours.
    const f = clamp(0.05 + 0.08 * (lastRate / 2), 0.05, 0.13);
    const multiplier = (1 + f * (jump - 1)) / jump;
    const medianRate = clampPct(lastRate * multiplier);
    const rare = lastRate < 0.5 ? 0.2 : lastRate < 2 ? 0.12 : 0;
    const sigma = Math.sqrt(
      0.45 * 0.45 + Math.pow(0.55 * Math.sqrt(horizonDays), 2) + 0.15 * 0.15 + rare * rare,
    );
    const m = logit(medianRate);
    const q = Z.map((z) => round4(clampPct(invLogit(m + z * sigma))));
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
}

/**
 * Real history: fit the population and each trophy's earners in log-log time,
 * and the rate's logit in log time, over the recent window; average the two
 * median views in logit space.
 */
function fittedForecast(
  snaps: Snapshot[],
  trophies: Trophy[],
  releaseMs: number,
  targetMs: number,
  horizonMinutes: number,
): Array<{ id: number; q: number[] }> {
  const last = snaps[snaps.length - 1];
  const lastMs = last.ms;
  // Recent window: the last 24 hours of distinct snapshots, widened until
  // it holds at least 3 snapshots (or everything, if that is all there is).
  let window = snaps.filter((s) => s.ms >= lastMs - DAY_H * 3_600_000);
  if (window.length < 3) window = snaps.slice();
  const xWin = window.map((s) => logTime(s.ms, releaseMs));
  const xTarget = logTime(targetMs, releaseMs);
  const horizonDays = horizonMinutes / 1440;

  // Population fit, slope shrunk toward the launch-growth prior 0.55.
  const logP = window.map((s) => Math.log(Math.max(1, s.players)));
  const pFit = ols(xWin, logP);
  const bP = clamp(shrinkSlope(pFit.b, 0.55, window.length), 0.05, 1.5);
  const aP = logP[logP.length - 1] - bP * xWin[xWin.length - 1];
  const pTarget = clamp(
    Math.exp(aP + bP * xTarget),
    last.players,
    last.players * 12,
  );

  return trophies.map((t, i) => {
    const lastRateRaw = last.rates[i];
    const lastRate = typeof lastRateRaw === 'number' ? lastRateRaw : 1;
    if (lastRate >= HI) {
      return { id: t.id, q: [99.9, 99.98, 99.99, 99.99, 99.99] };
    }
    // Earners series for this trophy over the window (skip null reads).
    const xs: number[] = [];
    const logE: number[] = [];
    const logR: number[] = [];
    for (let k = 0; k < window.length; k++) {
      const r = window[k].rates[i];
      if (typeof r !== 'number') continue;
      xs.push(xWin[k]);
      logE.push(Math.log((r * window[k].players) / 100 + 0.5));
      logR.push(logit(r));
    }
    let medianLogit: number;
    let resid = 0.1;
    if (xs.length >= 2) {
      // Earners fit: slope shrunk toward the population slope.
      const eFit = ols(xs, logE);
      const bE = clamp(shrinkSlope(eFit.b, bP, xs.length), 0, bP + 1.2);
      const aE = logE[logE.length - 1] - bE * xs[xs.length - 1];
      const eTarget = Math.max(0.5, Math.exp(aE + bE * xTarget) - 0.5);
      const rateDecomp = clampPct((eTarget / pTarget) * 100);
      // Rate-logit fit: slope shrunk toward 0 (persistence).
      const rFit = ols(xs, logR);
      const bR = clamp(shrinkSlope(rFit.b, 0, xs.length), -2.5, 3);
      const aR = logR[logR.length - 1] - bR * xs[xs.length - 1];
      const rateTrend = clampPct(invLogit(aR + bR * xTarget));
      resid = Math.max(rFit.resid, eFit.resid * 0.5);
      medianLogit = (logit(rateDecomp) + logit(rateTrend)) / 2;
    } else {
      // Too little per-trophy history: hold the last rate.
      medianLogit = logit(lastRate);
    }
    // A rate may not move more than 3.5 logit units from its last read.
    const lastLogit = logit(lastRate);
    medianLogit = clamp(medianLogit, lastLogit - 3.5, lastLogit + 3.5);
    const rare = lastRate < 0.5 ? 0.2 : lastRate < 2 ? 0.12 : 0;
    const sigma = Math.sqrt(
      0.3 * 0.3 +
        Math.pow(0.55 * Math.sqrt(horizonDays), 2) +
        Math.pow(clamp(resid, 0.1, 0.6), 2) +
        rare * rare,
    );
    const q = Z.map((z) => round4(clampPct(invLogit(medianLogit + z * sigma))));
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
}

/** Persistence fallback: last rate with a random-walk spread. Never invalid. */
function fallbackForecast(
  lastRates: Array<number | null>,
  trophies: Trophy[],
  horizonMinutes: number,
): Array<{ id: number; q: number[] }> {
  const spread = 0.5 * Math.sqrt(horizonMinutes / 1440);
  return trophies.map((t, i) => {
    const r = typeof lastRates[i] === 'number' ? (lastRates[i] as number) : 1;
    if (r >= HI) return { id: t.id, q: [99.9, 99.98, 99.99, 99.99, 99.99] };
    const m = logit(r);
    const q = Z.map((z) => round4(clampPct(invLogit(m + z * spread))));
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const checkpointMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  const snaps = snapshotsOf(c.history);
  if (snaps.length === 0) {
    // No history at all: a wide, low forecast rather than a crash.
    return c.trophies.map((t) => ({ id: t.id, q: [0.05, 0.3, 1, 3, 10] }));
  }
  const releaseMs = c.game?.release_date
    ? Date.parse(`${c.game.release_date}T00:00:00Z`)
    : snaps[0].ms;
  try {
    if (snaps.length === 1) {
      return flatForecast(snaps, c.trophies, checkpointMs, c.horizon_minutes);
    }
    return fittedForecast(snaps, c.trophies, releaseMs, targetMs, c.horizon_minutes);
  } catch {
    const last = snaps[snaps.length - 1];
    return fallbackForecast(last.rates, c.trophies, c.horizon_minutes);
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
