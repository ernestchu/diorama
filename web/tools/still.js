// Still frames of a drive from the page, headless: for each frame and view, the scene alone at WxH (device pixel
// ratio 1) as a PNG, and the engine's snapshot of it as JSON. The page's half of the WebGL / Metal parity check.
//
//   python -m dioramaweb view drive/ --no-browser &
//   node still.js http://127.0.0.1:8642/ out/ 3600,3700 [road,bird] [390x844] [dark]
//
// -> out/<frame>-<view>.png, out/<frame>-<view>.json. CHROME: the browser (default /usr/bin/google-chrome).
const puppeteer = require('puppeteer-core');
const fs = require('fs');
(async () => {
  const [base, outDir, frames, views = 'road,bird', size = '390x844', theme = 'dark'] = process.argv.slice(2);
  const [w, h] = size.split('x').map(Number);
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new',
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: w, height: h, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.log('pageerror:', e.message));
  let failed = 0;
  for (const f of frames.split(',').map(Number)) {
    for (const v of views.split(',')) {
      await page.goto(`${base.replace(/\/$/, '')}/diorama.html?drive=drive.diorama&still=${f}&view=${v}&theme=${theme}`);
      await page.waitForFunction(() => document.title.startsWith('still '), { timeout: 120000 });
      const title = await page.title();
      if (title !== 'still ready') { console.log(`${f} ${v}: ${title}`); failed++; continue; }
      await page.screenshot({ path: `${outDir}/${f}-${v}.png` });
      fs.writeFileSync(`${outDir}/${f}-${v}.json`, JSON.stringify(await page.evaluate(() => window.dioramaStill)));
      console.log(`${f} ${v}`);
    }
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
