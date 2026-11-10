"""Label the mined approach frames (mine_auto.py) with Claude Haiku through the Message Batches API (half price).

The system prompt is vlm/prompt.md, the labelling guideline written for the vision-language model. vlm/shots.json
holds eight worked examples: hand-labelled training frames by id, with the answer and why. Their images are rendered
from the built frames ($STOPLINE_C1M/yuv, build_c1m.py) and uploaded once with the Files API, so a request carries
only its own frame (a batch is limited to 256 MB). The instructions and examples form a cached prefix: a 1-hour cache
breakpoint on the last example's answer, since a batch's requests run over minutes to an hour.

  python vlm_label.py submit   -> $STOPLINE_C1M/auto/batches.json (batch ids, the uploaded files)
  python vlm_label.py fetch    -> $STOPLINE_C1M/auto/vlm.json (frame id -> [x1, y1, x2, y2] in model-frame px, or
                                  null), auto/vlm_usage.json, and the token use and cost
Reads ANTHROPIC_API_KEY. Labelling the 3467 frames of the paper cost $0.37.
"""
import base64, json, os, re, sys
import numpy as np, cv2
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from paths import C1M

MODEL = 'claude-haiku-5-5'
HERE = os.path.dirname(os.path.abspath(__file__))
AUTO = f'{C1M}/auto'
STATE = f'{AUTO}/batches.json'
CHUNK = 2000  # requests per batch: ~75 KB each, well under 256 MB
# $ per million tokens at batch prices (half of 0.10 / 0.50; cache read 0.1x, 1-hour cache write 2x the input price)
PRICE = {'input': 0.05, 'output': 0.25, 'cache_read': 0.005, 'cache_write': 0.10}


def rgb2x(path):
  """A built frame (Y, U, V png at 2x, BT.601 video range) as a 1024x512 BGR image."""
  yuv = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
  y, u, v = yuv[..., 0] - 16, yuv[..., 1] - 128, yuv[..., 2] - 128
  out = np.stack([1.164 * y + 2.017 * u, 1.164 * y - 0.392 * u - 0.813 * v, 1.164 * y + 1.596 * v], -1)
  return np.clip(out, 0, 255).astype(np.uint8)


def b64(data):
  return base64.standard_b64encode(data).decode()


def submit(client):
  prompt = open(f'{HERE}/vlm/prompt.md').read()
  shots = json.load(open(f'{HERE}/vlm/shots.json'))
  files = {}
  for e in shots:
    jpg = cv2.imencode('.jpg', rgb2x(f"{C1M}/yuv/{e['id']}.png"), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    files[e['id']] = client.files.upload(file=(f"{e['id']}.jpg", jpg, 'image/jpeg')).id
  prefix = []
  for e in shots:
    line = e['answer']['line']
    if line and line[0] > line[2]:
      line = line[2:] + line[:2]
    text = {'type': 'text', 'text': f'{e["why"]}\n{json.dumps({"line": line})}'}
    if e is shots[-1]:
      text['cache_control'] = {'type': 'ephemeral', 'ttl': '1h'}
    prefix += [{'role': 'user', 'content': [{'type': 'image', 'source': {'type': 'file', 'file_id': files[e['id']]}},
                                            {'type': 'text', 'text': 'Label this image.'}]},
               {'role': 'assistant', 'content': [text]}]
  names = sorted(json.load(open(f'{AUTO}/items.json')))
  ids = []
  for s in range(0, len(names), CHUNK):
    reqs = [Request(custom_id=n, params=MessageCreateParamsNonStreaming(
              model=MODEL, max_tokens=4000, system=prompt, output_config={'effort': 'medium'},
              messages=prefix + [{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                             'data': b64(open(f'{AUTO}/rgb/{n}.jpg', 'rb').read())}},
                {'type': 'text', 'text': 'Label this image.'}]}]))
            for n in names[s:s + CHUNK]]
    b = client.messages.batches.create(requests=reqs)
    ids.append(b.id)
    print('batch', b.id, len(reqs), b.processing_status, flush=True)
  json.dump({'batches': ids, 'files': files, 'n': len(names)}, open(STATE, 'w'), indent=1)


def fetch(client):
  st = json.load(open(STATE))
  for bid in st['batches']:
    b = client.messages.batches.retrieve(bid)
    print(bid, b.processing_status, b.request_counts)
    if b.processing_status != 'ended':
      return
  preds, usage, bad = {}, {}, 0
  for bid in st['batches']:
    for r in client.messages.batches.results(bid):
      if r.result.type != 'succeeded':
        bad += 1; continue
      msg = r.result.message
      text = ''.join(c.text for c in msg.content if c.type == 'text')
      m = re.findall(r'\{[^{}]*"line"[^{}]*\}', text)
      try:
        line = json.loads(m[-1])['line']
        preds[r.custom_id] = [round(float(v) / 2, 2) for v in line] if line else None  # 1024x512 -> model frame
      except Exception:
        preds[r.custom_id] = None; bad += 1
      u = msg.usage
      usage[r.custom_id] = {'input': u.input_tokens, 'output': u.output_tokens,
                            'cache_read': u.cache_read_input_tokens or 0, 'cache_write': u.cache_creation_input_tokens or 0}
  json.dump(preds, open(f'{AUTO}/vlm.json', 'w'))
  json.dump(usage, open(f'{AUTO}/vlm_usage.json', 'w'))
  tot = {k: sum(v[k] for v in usage.values()) for k in PRICE}
  cost = sum(tot[k] * PRICE[k] for k in PRICE) / 1e6
  print(len(preds), 'frames,', sum(p is not None for p in preds.values()), 'with a line,', bad, 'failed or unparsed')
  print('tokens', tot, f'cost ${cost:.3f} (${cost / max(1, len(usage)):.5f} a frame)')


if __name__ == '__main__':
  {'submit': submit, 'fetch': fetch}[sys.argv[1]](anthropic.Anthropic())
