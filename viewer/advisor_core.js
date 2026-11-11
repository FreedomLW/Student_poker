// Advisor core: house rules (trump always ♦, joker = 7♠: lead only "highest of suit X", follow as joker-high or as
// plain 7♠; must follow / must trump; dealer hook from `hookMin` cards), observation encoding identical to
// env/features.py (feature set v1) and MLP inference. Shared by viewer/advisor.html and the Node parity test.
(function (root) {
  "use strict";
  const NR = 9, NC = 36, JOKER = 1;            // 7♠ = suit 0 (S) * 9 + rank 1 (7)
  const TRUMP = 2;                              // diamonds
  const A_J_FOLLOW_HIGH = 36, A_J_FOLLOW_LOW = 37, A_J_LEAD_HIGH = 38, A_J_LEAD_LOW = 42, BID_OFFSET = 46, N_ACTIONS = 59;
  const K_NORMAL = 0, K_JHIGH = 1, K_JLOW = 2;
  const suitOf = c => Math.floor(c / NR), rankOf = c => c % NR;

  function baseDim(n, mc) { return 36 * 3 + (n - 1) * 36 + (n - 1) * 2 + 6 + 5 + mc + 2 + n * 7 + n * 2 + (n - 1) * 5 + 4 + 4 + 2; }

  // ---------------------------------------------------------------- state (replayed from an action log)
  // setup = {n, k, dealer, me, myHand:[card ids], dark, hookMin}
  function newState(setup) {
    const n = setup.n;
    return {
      n, k: setup.k, dealer: setup.dealer, first: (setup.dealer + 1) % n, me: setup.me, dark: !!setup.dark,
      hookMin: setup.hookMin == null ? 4 : setup.hookMin, maxCards: Math.floor(36 / n), trump: TRUMP,
      myHand: new Set(setup.myHand), bids: Array(n).fill(-1), nBids: 0, won: Array(n).fill(0),
      played: new Set(), tP: [], tC: [], tK: [], lead: -1, demand: false, leader: (setup.dealer + 1) % n,
      cur: (setup.dealer + 1) % n, phase: 0, tricks: 0, voids: Array(n).fill(0), history: [],
    };
  }
  function hookApplies(s) { return s.k >= s.hookMin; }
  function legalBids(s) {
    const out = [];
    for (let b = 0; b <= s.k; b++) out.push(b);
    if (hookApplies(s) && s.nBids === s.n - 1) {
      const sb = s.bids.reduce((a, b) => a + (b >= 0 ? b : 0), 0);
      const f = s.k - sb;
      const i = out.indexOf(f); if (i >= 0) out.splice(i, 1);
    }
    return out.map(b => BID_OFFSET + b);
  }
  // legal actions for ME (my hand is known)
  function legalMine(s) {
    if (s.phase === 0) return legalBids(s);
    if (s.phase === 2) return [];
    const hand = [...s.myHand].sort((a, b) => a - b);
    const hasJ = s.myHand.has(JOKER);
    const plain = hand.filter(c => c !== JOKER);
    if (!s.tP.length) {
      const acts = plain.slice();
      if (hasJ) for (let x = 0; x < 4; x++) acts.push(A_J_LEAD_HIGH + x);   // house rules: no low joker lead
      return acts;
    }
    const L = s.lead;
    const inL = plain.filter(c => suitOf(c) === L);
    let acts;
    if (inL.length) acts = s.demand ? [inL[inL.length - 1]] : inL;
    else {
      const tm = plain.filter(c => suitOf(c) === s.trump);
      acts = tm.length ? tm : plain;
    }
    acts = acts.slice();
    if (hasJ) acts.push(A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW);
    return acts;
  }
  // actions another player may have taken (unknown hand): any unseen card or joker variant
  function possibleOther(s) {
    if (s.phase === 0) return legalBids(s);
    const unseen = [];
    for (let c = 0; c < NC; c++) if (!s.myHand.has(c) && !s.played.has(c) && c !== JOKER) unseen.push(c);
    const jokerFree = !s.myHand.has(JOKER) && !s.played.has(JOKER);
    if (!s.tP.length) return unseen.concat(jokerFree ? [0, 1, 2, 3].map(x => A_J_LEAD_HIGH + x) : []);
    return unseen.concat(jokerFree ? [A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW] : []);
  }
  function trickWinner(s) {
    let bi = 0, bk = -1;
    for (let i = 0; i < s.tP.length; i++) {
      const kind = s.tK[i];
      let k;
      if (kind === K_JHIGH) k = 1000 + i;
      else if (kind === K_JLOW) k = i === 0 ? (s.lead === s.trump ? 100 : 10) : 0;
      else {
        const c = s.tC[i], su = suitOf(c);
        k = su === s.trump ? 101 + rankOf(c) : (su === s.lead ? 11 + rankOf(c) : 0);
      }
      if (k > bk) { bk = k; bi = i; }
    }
    return s.tP[bi];
  }
  function apply(s, a) {
    const p = s.cur, n = s.n;
    if (s.phase === 0) {
      s.bids[p] = a - BID_OFFSET; s.nBids++;
      if (s.nBids === n) { s.phase = 1; s.cur = s.leader; } else s.cur = (p + 1) % n;
      return;
    }
    const leading = !s.tP.length;
    let c, kind;
    if (a < 36) {
      c = a; kind = K_NORMAL;
      if (leading) { s.lead = suitOf(c); s.demand = false; }
      else {
        const su = suitOf(c);
        if (su !== s.lead) {
          s.voids[p] |= 1 << s.lead;
          if (s.trump >= 0 && su !== s.trump) s.voids[p] |= 1 << s.trump;
        }
      }
    } else {
      c = JOKER;
      if (a === A_J_FOLLOW_HIGH) kind = K_JHIGH;
      else if (a === A_J_FOLLOW_LOW) kind = K_NORMAL;            // house rule: joker played as the plain 7♠
      else if (a < A_J_LEAD_LOW) { kind = K_JHIGH; s.lead = a - A_J_LEAD_HIGH; s.demand = true; }
      else { kind = K_JLOW; s.lead = a - A_J_LEAD_LOW; s.demand = false; }
    }
    if (p === s.me) s.myHand.delete(c);
    s.played.add(c);
    s.tP.push(p); s.tC.push(c); s.tK.push(kind);
    if (s.tP.length < n) { s.cur = (p + 1) % n; return; }
    const w = trickWinner(s);
    s.won[w]++;
    s.history.push({leader: s.leader, cards: s.tC.slice(), kinds: s.tK.slice(), lead: s.lead, winner: w, players: s.tP.slice()});
    s.tricks++; s.tP = []; s.tC = []; s.tK = []; s.lead = -1; s.demand = false; s.leader = w; s.cur = w;
    if (s.tricks === s.k) s.phase = 2;
  }
  function replay(setup, log) { const s = newState(setup); for (const a of log) apply(s, a); return s; }

  // ---------------------------------------------------------------- encoding == env/features.py (_encode_base)
  function encode(s) {
    const n = s.n, k = s.k, mc = s.maxCards, me = s.me;
    const x = new Float32Array(baseDim(n, mc));
    const hidden = s.dark && s.phase === 0;
    const hand = hidden ? [] : [...s.myHand];
    let i = 0;
    for (const c of hand) x[i + c] = 1;
    i += 36;
    for (const c of s.played) x[i + c] = 1;
    i += 36;
    const hs = new Set(hand);
    for (let c = 0; c < 36; c++) if (!hs.has(c) && !s.played.has(c)) x[i + c] = 1;
    i += 36;
    const tb = i, jb = i + (n - 1) * 36;
    for (let t = 0; t < s.tP.length; t++) {
      const rel = ((s.tP[t] - me) % n + n) % n;
      if (rel === 0) continue;
      x[tb + (rel - 1) * 36 + s.tC[t]] = 1;
      if (s.tK[t] === K_JHIGH) x[jb + (rel - 1) * 2] = 1;
      else if (s.tK[t] === K_JLOW) x[jb + (rel - 1) * 2 + 1] = 1;
    }
    i = jb + (n - 1) * 2;
    x[i + (s.lead >= 0 ? s.lead : 4)] = 1; i += 5;
    x[i] = s.demand ? 1 : 0; i += 1;
    x[i + (s.trump >= 0 ? s.trump : 4)] = 1; i += 5;
    x[i + k - 1] = 1; i += mc;
    x[i] = (k - s.tricks) / k; x[i + 1] = s.tricks / k; i += 2;
    for (let rel = 0; rel < n; rel++) {
      const p = (me + rel) % n, b = s.bids[p], base = i + rel * 7;
      if (b >= 0) {
        const need = b - s.won[p];
        x[base] = b / k; x[base + 2] = need / k; x[base + 3] = need > 0 ? 1 : 0;
        x[base + 4] = need === 0 ? 1 : 0; x[base + 5] = need < 0 ? 1 : 0; x[base + 6] = 1;
      }
      x[base + 1] = s.won[p] / k;
    }
    i += n * 7;
    x[i + ((me - s.first) % n + n) % n] = 1; i += n;
    x[i + ((s.dealer - me) % n + n) % n] = 1; i += n;
    for (let rel = 1; rel < n; rel++) {
      const p = (me + rel) % n, v = s.voids[p], base = i + (rel - 1) * 5;
      for (let su = 0; su < 4; su++) if ((v >> su) & 1) x[base + su] = 1;
      if (s.trump >= 0 && ((v >> s.trump) & 1)) x[base + 4] = 1;
    }
    i += (n - 1) * 5;
    x[i] = s.dark ? 1 : 0;
    x[i + 1] = s.phase === 0 ? 1 : 0;
    let sb = 0, nb = 0;
    for (const b of s.bids) if (b >= 0) { sb += b; nb++; }
    x[i + 2] = sb / k;
    x[i + 3] = (nb === n - 1 && hookApplies(s) && k - sb >= 0 && k - sb <= k) ? (k - sb) / k : -1;
    i += 4;
    for (let su = 0; su < 4; su++) x[i + su] = hand.filter(c => c !== JOKER && suitOf(c) === su).length / k;
    i += 4;
    x[i] = hs.has(JOKER) ? 1 : 0;
    x[i + 1] = s.played.has(JOKER) ? 1 : 0;
    return x;
  }

  // ---------------------------------------------------------------- MLP (ReLU), weights as {layers:[{W:Float32Array(out*in), b, in, out}]}
  function f16ToF32(u16) {
    const out = new Float32Array(u16.length);
    for (let j = 0; j < u16.length; j++) {
      const h = u16[j], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[j] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : (e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024));
    }
    return out;
  }
  function b64ToU16(b64) {
    const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    const u8 = new Uint8Array(bin.length);
    for (let j = 0; j < bin.length; j++) u8[j] = bin.charCodeAt(j);
    return new Uint16Array(u8.buffer);
  }
  function loadNet(spec) {
    return {layers: spec.layers.map(L => ({in: L.in, out: L.out, W: f16ToF32(b64ToU16(L.W)), b: f16ToF32(b64ToU16(L.b))}))};
  }
  function forward(net, x) {
    let h = x;
    net.layers.forEach((L, li) => {
      const y = new Float32Array(L.out);
      for (let o = 0; o < L.out; o++) {
        let acc = L.b[o];
        const row = o * L.in;
        for (let j = 0; j < L.in; j++) { const v = h[j]; if (v !== 0) acc += L.W[row + j] * v; }
        y[o] = li < net.layers.length - 1 ? (acc > 0 ? acc : 0) : acc;
      }
      h = y;
    });
    return h;
  }
  function qValues(model, s) {
    const net = s.phase === 0 ? model.bid : model.play;
    return forward(net, encode(s));
  }

  const API = {NR, NC, JOKER, TRUMP, A_J_FOLLOW_HIGH, A_J_FOLLOW_LOW, A_J_LEAD_HIGH, A_J_LEAD_LOW, BID_OFFSET, N_ACTIONS,
    suitOf, rankOf, newState, replay, apply, legalMine, possibleOther, legalBids, trickWinner, encode, loadNet, forward, qValues, baseDim};
  if (typeof module !== "undefined" && module.exports) module.exports = API; else root.Advisor = API;
})(typeof window !== "undefined" ? window : globalThis);
