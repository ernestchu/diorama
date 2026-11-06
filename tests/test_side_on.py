"""The lead filter for vehicles seen side-on (dioramaweb.stopline.side_on), which the app's StopLineDecoder mirrors."""
from dioramaweb.stopline import side_on


def test_near_vehicles_pass_whatever_their_shape():
  assert not side_on(0, 0, 512, 254)        # fills the frame, cut by its bottom
  assert not side_on(20, 0, 500, 240)       # 480 x 240 at 6.4 m: clipped by the frame, not side-on
  assert not side_on(150, 100, 350, 200)    # a rear at 7.5 m


def test_side_on_vehicles_further_out_are_dropped():
  assert side_on(100, 60, 300, 130)         # 200 x 70 at 13.8 m: wider than 1.9 x its height
  assert side_on(300, 90, 512, 139)         # 212 px at 12.4 m: 2.9 m on the road, aspect 4.3
  assert side_on(200, 60, 330, 100)         # 130 px at 21.7 m: 3.1 m, wider than any rear


def test_rears_further_out_pass():
  assert not side_on(220, 70, 290, 115)     # a car's rear at 16.9 m: 1.3 m wide
