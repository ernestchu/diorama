"""The lead-car teacher: D-FINE (Apache-2.0) from Hugging Face, COCO vehicles merged into one class."""
import torch
from transformers import AutoImageProcessor, DFineForObjectDetection

MODEL = 'ustc-community/dfine-xlarge-obj2coco'
VEHICLES = {'car', 'motorcycle', 'bus', 'truck', 'bicycle', 'train'}


class Teacher:
  def __init__(self, device='cuda', threshold=0.5):
    self.proc = AutoImageProcessor.from_pretrained(MODEL)
    self.model = DFineForObjectDetection.from_pretrained(MODEL).to(device).eval().half()
    self.device, self.threshold = device, threshold
    names = self.model.config.id2label
    self.keep = {i for i, n in names.items() if n in VEHICLES}

  @torch.no_grad()
  def __call__(self, images):
    """RGB uint8 HxWx3 arrays -> per image a list of (x1, y1, x2, y2, score) vehicle boxes in pixels."""
    inp = self.proc(images=images, return_tensors='pt').to(self.device)
    out = self.model(pixel_values=inp['pixel_values'].half())
    sizes = [im.shape[:2] for im in images]
    res = self.proc.post_process_object_detection(out, target_sizes=sizes, threshold=self.threshold)
    return [[(*b.tolist(), s.item()) for b, s, l in zip(r['boxes'], r['scores'], r['labels']) if l.item() in self.keep]
            for r in res]
