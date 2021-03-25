DEPTH_LAYERS = 50
POSE_LAYERS = 18
FRAME_IDS = [0, -1, 1]
HEIGHT = 320
WIDTH = 576

model = dict(
    depth_num_layers=DEPTH_LAYERS,
    pose_num_layers=POSE_LAYERS,
    frame_ids=FRAME_IDS,
    height=HEIGHT,
    width=WIDTH,
)
