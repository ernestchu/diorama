These files are copied from [jetlink](https://github.com/zoompilot/jetlink)
(`main`), by Zeph Leggett, under the MIT License. Only their imports are changed,
to point at this folder.

- `protocol.py`: the link's wire format, the one zoompilot speaks.
- `spec.py`, `onnx_meta.py`: a driving model's sizes, read from its ONNX.
- `queues.py`: the driving model's input history, as modeld keeps it.
