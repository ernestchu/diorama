// The engine over a whole drive, without a browser: every frame's DioramaEngine.snapshot, one JSON object per line.
// The page's half of the scene parity check (the app's DioramaScene must give the same states).
//
//   node scene.js drive/drive.diorama > scene.jsonl
const fs = require('fs');
const E = require('../page/diorama-engine.js');
const buf = fs.readFileSync(process.argv[2]);
const engine = new E.Engine();
for (const m of E.readFile(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))) {
  const before = engine.state;
  engine.push(m);
  if (m.kind === 'D' && engine.state && engine.state !== before) process.stdout.write(JSON.stringify(E.snapshot(engine.state)) + '\n');
}
