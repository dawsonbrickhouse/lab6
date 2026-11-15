import sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sorted(glob.glob(f"my_logs/{sys.argv[1]}/version_*")):
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    v = ea.Scalars("val_acc") if "val_acc" in ea.Tags()["scalars"] else []
    if v:
        best = max(v, key=lambda e: e.value)
        print(d, "epochs:", len(v), "last:", round(v[-1].value, 4), "best:", round(best.value, 4), "@epoch", v.index(best) + 1, "elapsed:", round(v[-1].wall_time - ea.FirstEventTimestamp()), "s")
