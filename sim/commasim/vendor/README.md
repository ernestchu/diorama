These files are copied from [jetlink](https://github.com/zoompilot/jetlink)
(`main`), by Zeph Leggett, under the MIT License. Only their imports are changed,
to point at this folder.

- `protocol.py`: the link's wire format, the one zoompilot speaks.
- `spec.py`, `onnx_meta.py`: a driving model's sizes, read from its ONNX.
- `queues.py`: the driving model's input history, as modeld keeps it.

`comma_ui/model_renderer.py` is the comma 4 display's model drawing, from
[zoompilot](https://github.com/ernestchu/zoompilot/tree/ernestchu/diorama) at
commit `2ef7f8522f35a91e017b102673ec31fb9f1cebe7`
(`openpilot/selfdrive/ui/mici/onroad/model_renderer.py`), under the MIT License,
unchanged. `sim/comma_ui_parity.py` runs it to make the fixtures that hold the
app's drawing of the comma's prediction to it. To move the pin, replace the file,
and change the commit here, in `comma_ui_parity.py` and in `CommaUI.swift`.
