import math

def calculate_speed(prev_position, curr_position, frame_interval):
    """Calculate speed between two positions in meters per second."""
    distance = math.sqrt((curr_position[0] - prev_position[0]) ** 2 + (curr_position[1] - prev_position[1]) ** 2)
    speed = distance / frame_interval
    return speed
