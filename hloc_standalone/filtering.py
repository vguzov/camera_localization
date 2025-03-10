import numpy as np

def none_interp(path_dict):
    names = sorted(path_dict.keys())
    path = [path_dict[n] for n in names]
    new_path_dict = {}
    for ind in range(len(path)):
        frame = path[ind]
        if frame is None:
            prev_diff = 0
            next_diff = 0
            prev_frame = None
            next_frame = None
            for diff in range(1,ind+1):
                if path[ind-diff] is not None:
                    prev_frame = path[ind-diff]
                    prev_diff = diff
                    break
            for diff in range(1, len(path)-ind):
                if path[ind+diff] is not None:
                    next_frame = path[ind+diff]
                    next_diff = diff
                    break
            if prev_frame is not None and next_frame is not None:
                coeffs = np.array([[next_diff], [prev_diff]], dtype=np.float)/(prev_diff+next_diff)
                frame = {k:(np.array([prev_frame[k],next_frame[k]])*coeffs).sum(0) for k in prev_frame.keys()}
        new_path_dict[names[ind]] = frame
    return new_path_dict

def velocity_filter(path_dict, velocity_thresh = 0.05):
    names = sorted(path_dict.keys())
    path = [path_dict[n] for n in names]
    new_path_dict = {}
    vels = []
    for ind in range(len(path)):
        frame = path[ind]
        if ind>0 and ind<len(path)-1:
            prev_frame = path[ind-1]
            next_frame = path[ind+1]
            if frame is None or prev_frame is None or next_frame is None:
                frame = None
            else:
                prev_velocity = np.sqrt(((np.array(frame['position']) - np.array(prev_frame['position']))**2).sum())
                next_velocity = np.sqrt(((np.array(frame['position']) - np.array(next_frame['position']))**2).sum())
                vel = (prev_velocity+next_velocity)/2.
                vels.append(vel)
                if vel>velocity_thresh:
                    frame = None
        new_path_dict[names[ind]] = frame
    return new_path_dict, np.array(vels)