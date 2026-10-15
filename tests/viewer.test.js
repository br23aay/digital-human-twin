// Viewer smoke test. Needs Node + Playwright and a local server on port 8000 in the app folder:
//   cd app && python -m http.server 8000      then: node tests/viewer.test.js
const { chromium } = require('playwright');
const URL = process.env.TWIN_URL || 'http://localhost:8000/my-twin.html';
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
  let failures = 0; const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) failures++; };
  for (const vp of [{ width: 1400, height: 900 }, { width: 420, height: 860 }]) {
    const p = await b.newPage({ viewport: vp }); const errs = [];
    p.on('pageerror', e => errs.push(e.message));
    await p.goto(URL); await p.waitForFunction(() => window.__twin && window.__twin.ready, { timeout: 120000 });
    await p.evaluate(() => window.__setAutoRotate(false));
    const tag = `${vp.width}x${vp.height}`;
    for (const model of ['v3', 'v2', 'v1']) {
      await p.click(`[data-model=${model}]`); await p.waitForFunction(m => window.__twin && window.__twin.ready && window.__twin.model === m, model, { timeout: 120000 });
      for (const view of ['body', 'face']) {
        await p.click(`[data-v=${view}]`); await p.waitForTimeout(400);
        const pr = await p.evaluate(() => window.__projected());
        ok(pr && pr.minx > -1.05 && pr.maxx < 1.05 && pr.miny > -1.05 && pr.maxy < 1.05, `${tag} ${model} ${view}: framed inside viewport ${JSON.stringify(pr && Object.fromEntries(Object.entries(pr).map(([k, v]) => [k, +v.toFixed(2)])))}`);
        for (const mode of ['photo', 'holo', 'clay']) {
          await p.click(`[data-m=${mode}]`); await p.waitForTimeout(250);
          const lit = await p.evaluate(() => { const c = document.querySelector('canvas'); const g = document.createElement('canvas'); g.width = 64; g.height = 64; const x = g.getContext('2d'); x.drawImage(c, 0, 0, 64, 64); const d = x.getImageData(0, 0, 64, 64).data; let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i + 3] > 0 && d[i] + d[i + 1] + d[i + 2] > 40) n++; return n; });
          ok(lit > 20, `${tag} ${model} ${view} ${mode}: model visible (${lit} lit cells)`);
          if (model === 'v3' && vp.width === 1400) await p.screenshot({ path: `tests/shot_${model}_${view}_${mode}.png` });
        }
      }
    }
    // turntable toggle and zoom limits
    await p.click('#spin'); ok(await p.getAttribute('#spin', 'aria-pressed') === 'true', `${tag} turntable toggles on`);
    await p.mouse.move(vp.width / 2, vp.height / 2); for (let i = 0; i < 20; i++) await p.mouse.wheel(0, -400);
    await p.waitForTimeout(300); const pr = await p.evaluate(() => window.__projected());
    ok(pr !== null, `${tag} zoom keeps working`);
    ok(errs.length === 0, `${tag} no page errors ${errs.join('; ')}`);
    await p.close();
  }
  await b.close(); console.log(failures ? `${failures} FAILED` : 'ALL PASSED'); process.exit(failures ? 1 : 0);
})();
