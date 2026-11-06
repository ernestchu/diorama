// A stretch of a recorded drive in the Diorama viewer as numbered JPEGs, frame by frame: the footage seeked to each
// frame in turn (from 3 s before, so the scene has built up), the page left to draw it, then a screenshot.
//
//   node render_clip.js http://127.0.0.1:8642/ out/ FROM TO     (seconds into the drive `run` wrote)
//   ffmpeg -framerate 20 -i out/%05d.jpg -c:v libx264 -pix_fmt yuv420p clip.mp4
//
// CHROME: the browser to drive (default /usr/bin/google-chrome).
const puppeteer = require('puppeteer-core');
const fs = require('fs');
(async () => {
  const [url, outDir, from, to] = process.argv.slice(2);
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new',
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 560 });
  page.on('pageerror', (e) => console.log('pageerror:', e.message));
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => { const v = document.getElementById('footage'); return v && v.readyState >= 2; }, { timeout: 60000 });
  const f0 = Math.round(Number(from) * 20), f1 = Math.round(Number(to) * 20);
  for (let f = f0 - 60; f < f1; f++) {
    await page.evaluate((t) => new Promise((resolve) => {
      const v = document.getElementById('footage');
      v.pause();
      v.addEventListener('seeked', () => requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(resolve))), { once: true });
      v.currentTime = t;
    }), (f + 0.5) / 20);
    if (f >= f0) await page.screenshot({ path: `${outDir}/${String(f - f0).padStart(5, '0')}.jpg`, type: 'jpeg', quality: 90,
                                         clip: { x: 0, y: 42, width: 1600, height: 460 } });
  }
  await browser.close();
})();
