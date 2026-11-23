These files are copied from [jetlink](https://github.com/zoompilot/jetlink)
(`main`), by Zeph Leggett, under the MIT License. Only their imports are changed,
to point at this folder.

- `protocol.py`: the link's wire format, the one zoompilot speaks.
- `spec.py`, `onnx_meta.py`: a driving model's sizes, read from its ONNX.
- `queues.py`: the driving model's input history, as modeld keeps it.

`comma_ui/` is the comma 4 display's model drawing, from
[zoompilot](https://github.com/ernestchu/zoompilot) `main` at commit
`f628686b278675501933c8b9aba9efcb43534648`, each file at its path there, under the
MIT License, unchanged:

- `openpilot/selfdrive/ui/mici/onroad/model_renderer.py`: the comma 4's renderer.
- `openpilot/selfdrive/ui/onroad/model_renderer.py`: the base renderer it extends.
- `openpilot/selfdrive/ui/mici/onroad/status.py`: the status fade it draws through.

`sim/comma_ui_parity.py` runs them to make the fixtures that hold the app's drawing
of the comma's prediction to them, and its `PINNED` lists each file's sha256, which
zoompilot's tests check its own files against. To move the pin, replace the files,
and change the commit and hashes in `comma_ui_parity.py`, the commit here and in
`CommaUI.swift`; pin a commit on zoompilot's `main`, which is not rewritten.
