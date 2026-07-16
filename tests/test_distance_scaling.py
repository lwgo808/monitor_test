import math
from src.distance_display import distance_to_scale


def test_scale_increases_with_distance():
    s_near = distance_to_scale(30.0)
    s_ref = distance_to_scale(60.0)
    s_far = distance_to_scale(120.0)

    assert s_near < s_ref < s_far


def test_scale_clamps():
    # extremely small/negative distances fallback to base behavior
    assert distance_to_scale(-10) >= 0.3
    # extremely large distance clamps to max
    assert distance_to_scale(10000) <= 5.0


def test_known_mapping():
    # Using defaults: base_scale=0.8, ref=60 -> at 60cm scale ~= 0.8
    assert math.isclose(distance_to_scale(60.0), 0.8, rel_tol=1e-2)
