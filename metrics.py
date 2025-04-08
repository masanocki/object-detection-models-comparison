def calculate_fps_and_time(frame_times, total_time, frame_count):
    if not frame_times or frame_count == 0:
        return {"min_fps": 0, "avg_fps": 0, "max_fps": 0, "avg_frame_time": 0}

    true_avg_fps = frame_count / total_time

    frame_fpss = [1 / t for t in frame_times if t > 0]
    avg_frame_time = sum(frame_times) / len(frame_times)

    return {
        "min_fps": min(frame_fpss),
        "avg_fps": true_avg_fps,
        "max_fps": max(frame_fpss),
        "avg_frame_time": avg_frame_time,
    }
