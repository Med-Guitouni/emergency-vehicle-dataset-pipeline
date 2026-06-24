

### Detection and Tracking

The first step is finding vehicles in each frame. **YOLO** (version 8 extra-large) is an object detection neural network that looks at each image and draws a bounding box around every vehicle it finds, classifying it as car, truck, bus, or motorcycle. It does this independently for each frame ,it has no memory of what it saw one second ago.

Memory comes from **ByteTrack**, a tracking algorithm that takes YOLO's detections and links them across time. It does this by predicting where each vehicle should be in the next frame (using a Kalman Filter a mathematical model that estimates future position from current position and velocity) and then matching that prediction to the closest actual detection using bounding box overlap (IoU Intersection over Union, which measures how much two boxes overlap as a fraction of their combined area). When the overlap is high enough, ByteTrack decides it is the same vehicle and keeps the same ID number. This is what lets us say "truck 14 was here at second 30 and there at second 31"  without it, we would have a new unconnected list of anonymous boxes every second.

**EMAP** (Ego-Motion Aware Prediction) runs alongside ByteTrack to handle a specific problem: because the ambulance itself is moving, every vehicle in the image appears to drift backward even if it is stationary. EMAP uses the ego homography (a matrix describing how the whole image shifted between two frames due to the camera's own movement) and a rough depth estimate from **Depth Anything V2** (a neural network that produces a per-pixel relative depth map , not in real metres, just a ranking of what is closer and what is further) to subtract the camera's motion from each vehicle's predicted position before ByteTrack tries to match it. This way, a parked car on the shoulder looks stationary to the tracker instead of appearing to fly backward.

The **scene classifier** is a ResNet18 (a convolutional neural network with 18 layers, no transformers) trained on the MIT Places365 dataset . It looks at each frame and guesses the scene type eg highway, urban, intersection, or roundabout. This is used as a fallback for road type when a video has not been manually annotated, and it determines which lane width to use (highway lanes are wider than city lanes).

### From Pixels to Real-World Position

A camera only gives flat pictures. To know where a vehicle actually is on the road in metres, we use the fact that the camera sits at a known height above a flat road surface. The key insight is simple: a vehicle that is close to the ambulance appears low in the image (near the bottom), while a vehicle far away appears high in the image (near the horizon). The exact relationship is given by the pinhole camera model applied to a flat ground plane.

**Forward distance** (in metres):

```
y_forward = camera_height × focal_length / (bottom_y - horizon_row)
```

Where `bottom_y` is the pixel row of the bottom of the bounding box (where the vehicle's tyres touch the road), `horizon_row` is the pixel row where the road meets the sky in the distance (set at 55% of the frame height, tuned visually), `camera_height` is the height of the camera above the road (estimated at 1.4 metres), and `focal_length` is the camera's focal length in pixels (estimated as 80% of the frame width). The formula says: the number of pixels between the vehicle's tyres and the horizon is inversely proportional to the real distance. 
A vehicle 10 metres away sits many pixels below the horizon; a vehicle 50 metres away sits only a few pixels below. Dividing the camera height times the focal length by that pixel gap converts it to real metres.

**Lateral position** (how far left or right the vehicle is, in metres):

```
x_lateral = (bottom_x - image_centre) × y_forward / focal_length
```

Where `bottom_x` is the horizontal pixel position of the bottom-centre of the bounding box and `image_centre` is the middle column of the frame. This formula uses the forward distance we just computed: a vehicle that is 1 pixel to the right of centre at 10 metres is much closer laterally than a vehicle 1 pixel to the right at 50 metres.
Multiplying the pixel offset by the forward distance and dividing by the focal length converts horizontal pixels to real lateral metres. Positive means the vehicle is to the right of the ambulance, negative means to the left.

These two numbers  `x_meters` and `y_meters`  are the foundation. Every other metric in the pipeline is derived from them.

### Velocity: Splitting Speed Into Forward and Sideways

A single speed number hides the information we care about most. A vehicle pulling aside to let the ambulance pass moves sideways; a vehicle braking moves forward but slower.
To distinguish these, we split velocity into two components by looking at how the metric position changed between two consecutive seconds:

**Forward speed** (movement along the road):

```
forward_speed_ms = (y_now - y_previous) / dt
```

Positive means the vehicle is moving away from the ambulance (the gap is growing); negative means the ambulance is catching up (the gap is shrinking).
This is always relative to the ambulance , if both are travelling at the same real speed, forward_speed_ms reads approximately zero.

**Lateral speed** (movement across the road):

```
lateral_speed_ms = (x_now - x_previous) / dt
```

Positive means moving right, negative means moving left. This is the most direct signal for yielding: a vehicle pulling onto the right shoulder shows a clear positive lateral speed even when its forward speed is near zero.
This is why the primary yielding rule uses lateral speed ; it captures the exact motion that defines yielding.

**Overall speed** (combined magnitude):

```
speed_kmh = sqrt(forward² + lateral²) × 3.6
```

The square root combines both components into a single total speed, and multiplying by 3.6 converts from metres per second to kilometres per hour. 
This number tells you how fast the vehicle is moving overall but not in which direction - 
the split components above are more informative for behaviour analysis.

### Acceleration and Jerk

**Acceleration** measures how quickly the forward speed is changing:

```
acceleration = (forward_speed_now - forward_speed_previous) / dt
```

A negative value means the vehicle is decelerating relative to the ambulance (slowing down or the ambulance is speeding up). This is computed from the signed forward speed, not from the speed magnitude, because magnitude-based acceleration produced phantom braking events: when a vehicle's relative velocity passed through zero (the ambulance matching its speed exactly), the magnitude collapsed to zero and then rose again, looking like a hard brake followed by sudden acceleration even though nothing physical happened. Using the signed forward speed avoids this artifact entirely.

**Jerk** measures how suddenly the acceleration itself changes:

```
jerk = (acceleration_now - acceleration_previous) / dt
```

The difference between acceleration and jerk is the difference between braking and panic-braking. 
A vehicle that smoothly decelerates at 2 m/s² over several seconds has large acceleration but small jerk.
A vehicle that slams on the brakes :going from zero deceleration to heavy deceleration in one second  has large jerk. The annotator uses jerk alongside acceleration to catch panic-brake 
onsets that might straddle two 1-second frames and miss the pure acceleration threshold.

### Heading Angle
In work ( code that exists is dead )

### Trajectory Smoothing (RTS)

Before any velocity, acceleration, or jerk is computed, the raw positions are cleaned by a Rauch-Tung-Striebel (RTS) smoother. The problem it solves: YOLO's bounding box edge wobbles a few pixels every frame even on a perfectly stationary vehicle, and at 25 metres distance one pixel equals roughly one metre of forward distance, so far-away vehicles snap between discrete position values. Both effects produce fake velocities and accelerations.

A regular Kalman Filter only looks at the past as it processes each frame. The RTS smoother adds a second, backward pass over the entire finished trajectory: each position estimate is corrected using not only what came before it but also what came after it. The result is the smoothest possible trajectory that still fits all the observations. This is the same post-processing step that the highD and INTERACTION reference traffic datasets apply before publishing.

The smoother is also reliability-aware: measurements that came from clipped bounding boxes (flagged as `position_reliable = false`) are given 25 times more measurement noise in the filter, so the physics model (which assumes roughly constant velocity) dominates over the bad measurement. This is what tames trucks driving right beside the ambulance whose bounding boxes are cut off at the frame edge , their raw positions can jump ±3 metres per frame, but the smoother recognises those as unreliable and produces a smooth, physically plausible trajectory instead.

###  Bird's Eye View

All of these calculations produce a consistent real-world coordinate system: x in metres (lateral, left-right) and y in metres (forward, distance ahead), 
with the ambulance fixed at the origin (0, 0). 
