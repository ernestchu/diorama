You are labelling stop lines in images from a car's forward-facing road camera. Each image is 1024 pixels wide and
512 pixels tall; x runs from 0 (left) to 1024 (right) and y from 0 (top) to 512 (bottom). The camera looks straight
ahead along the car's direction of travel; the horizon of a level road is near y = 95, and the bottom edge of the
image shows the road about 5 m in front of the camera.

Task: decide whether the image shows the stop line the car would stop at, and if so give its two end points.

A stop line is a solid white line painted across the lane, where a car stops before an intersection, a stop sign, a
traffic light or a crosswalk. Label at most ONE line per image: the one across the EGO LANE, the lane the car is
driving in (normally the lane straight ahead in the middle of the image). Lines across other lanes, lines on the far
side of an intersection, crosswalk stripes, lane lines and other markings are not stop lines.

Rules:
0. Ego lane only. Label only the stop line across the lane the car is driving in, never the stop lines of other lanes,
   and only the part of it that lies across that lane. If only other lanes have a stop line, answer no line.
1. Distance. If the stop line is too far or too near, answer no line. Too far: the line is a thin sliver close to the
   horizon (its middle above about y = 135). A line in the distance across an intersection that is thin but clearly
   painted, with its middle below y = 135, counts. Too near: the line is low in the image, near the bottom edge (its middle
   below about y = 410), or cut off by the bottom of the image.
2. End points. Cut the line to the ego lane. Where the stop line meets the ego lane's lane lines or the road edge, put
   the end points at those intersections. Where the paint itself has clear ends inside the lane, put the end points at
   the ends of the paint. Put each end point on the middle of the line's thickness.
3. Paint only. If there is no visible stop-line paint and a stop line could only be inferred from context (an
   intersection ahead, a stop sign, cars waiting), answer no line. A line hidden by a vehicle in front is no line.

Answer with one JSON object and nothing else:
  {"line": [x1, y1, x2, y2]}    end points in pixels of this image, left end first
  {"line": null}                no stop line by these rules
