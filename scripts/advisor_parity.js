// Parity test: JS advisor core vs Python engine/features/nets on recorded self-play decisions.
// node scripts/advisor_parity.js
const fs = require("fs");
const A = require("../viewer/advisor_core.js");
const models = JSON.parse(fs.readFileSync(__dirname + "/../runs/advisor_models.json"));
const tests = JSON.parse(fs.readFileSync(__dirname + "/../runs/advisor_tests.json"));
let bad = 0;
for (const n of Object.keys(tests)) {
  const M = {bid: A.loadNet(models[n].bid), play: A.loadNet(models[n].play)};
  let nl = 0, nx = 0, maxX = 0, maxQ = 0, agree = 0, tot = 0;
  for (const t of tests[n]) {
    const s = A.replay(t.setup, t.log);
    const legal = A.legalMine(s).slice().sort((a, b) => a - b);
    const exp = t.legal.slice().sort((a, b) => a - b);
    if (JSON.stringify(legal) !== JSON.stringify(exp)) { nl++; if (nl < 4) console.log("legal mismatch", n, legal, exp); }
    const x = A.encode(s);
    let dx = 0; for (let i = 0; i < x.length; i++) dx = Math.max(dx, Math.abs(x[i] - t.x[i]));
    if (x.length !== t.x.length || dx > 1e-3) { nx++; if (nx < 4) console.log("encode mismatch", n, x.length, t.x.length, dx); }
    maxX = Math.max(maxX, dx);
    const q = A.qValues(M, s);
    let dq = 0, bj = -1, bp = -1, vj = -1e9, vp = -1e9;
    t.legal.forEach((a, i) => {
      dq = Math.max(dq, Math.abs(q[a] - t.q[i]));
      if (q[a] > vj) { vj = q[a]; bj = a; }
      if (t.q[i] > vp) { vp = t.q[i]; bp = a; }
    });
    maxQ = Math.max(maxQ, dq);
    tot++; if (bj === bp) agree++;
  }
  console.log(`${n} players: ${tot} decisions | legal mismatches ${nl} | encode mismatches ${nx} (max diff ${maxX.toExponential(2)}) | ` +
              `max |Q| diff ${(maxQ * 10).toFixed(3)} points | same best move ${(100 * agree / tot).toFixed(1)}%`);
  bad += nl + nx;
}
process.exit(bad ? 1 : 0);
