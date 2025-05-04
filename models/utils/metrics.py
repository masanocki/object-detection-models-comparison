def calculate_fps_and_time(frame_times, total_time, frame_count):
    if not frame_times or frame_count == 0:
        return {"min_fps": 0, "avg_fps": 0, "max_fps": 0, "avg_frame_time": 0}

    inference_fps = [1 / t for t in frame_times if t > 0]

    return {
        "min_fps": min(inference_fps),
        "avg_fps": sum(inference_fps) / len(inference_fps),
        "max_fps": max(inference_fps),
        "avg_frame_time": total_time / len(frame_times),
    }
