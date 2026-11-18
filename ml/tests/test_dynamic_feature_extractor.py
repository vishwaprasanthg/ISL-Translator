import numpy as np
import pytest

from ml.preprocessing.landmark_normalizer import create_feature_vector
from ml.preprocessing.dynamic_feature_extractor import (
    extract_dynamic_features,
    DYNAMIC_FEATURE_DIM,
    STATIC_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    HANDSHAPE_SLICE,
    LEFT_HANDSHAPE_SLICE,
    RIGHT_HANDSHAPE_SLICE,
    LEFT_WRIST_SLICE,
    LEFT_SCALE_IDX,
    RIGHT_WRIST_SLICE,
    RIGHT_SCALE_IDX,
    INTER_WRIST_VEC_SLICE,
    INTER_WRIST_DIST_IDX,
    LEFT_PRESENT_IDX,
    RIGHT_PRESENT_IDX,
)


def _make_dummy_hand(wrist_x=0.5, wrist_y=0.5, wrist_z=0.0, scale=0.1):
    """Creates a synthetic 21-landmark hand with wrist at (wrist_x, wrist_y, wrist_z) and landmarks proportional to scale."""
    landmarks = []
    for i in range(21):
        if i == 0:
            landmarks.append({'x': wrist_x, 'y': wrist_y, 'z': wrist_z})
        elif i == 9:
            landmarks.append({'x': wrist_x, 'y': wrist_y - scale, 'z': wrist_z})
        else:
            landmarks.append({
                'x': wrist_x + scale * 0.1 * (i % 5),
                'y': wrist_y - scale * 0.2 * (i // 5),
                'z': wrist_z + scale * 0.05 * i
            })
    return landmarks


def test_dynamic_feature_shape_and_dtype():
    hand = _make_dummy_hand()
    feat = extract_dynamic_features(right_hand_landmarks=hand)
    assert feat.shape == (140,)
    assert feat.dtype == np.float32
    assert len(feat) == DYNAMIC_FEATURE_DIM


def test_static_slice_parity():
    """Verify that indices 0..125 are identical to the existing static create_feature_vector."""
    left = _make_dummy_hand(wrist_x=0.3, scale=0.12)
    right = _make_dummy_hand(wrist_x=0.7, scale=0.14)

    static_feat = create_feature_vector(left, right)
    dyn_feat = extract_dynamic_features(left, right)

    np.testing.assert_allclose(dyn_feat[HANDSHAPE_SLICE], static_feat, rtol=1e-5, atol=1e-6)


def test_no_hands():
    feat = extract_dynamic_features(None, None)
    assert feat.shape == (140,)
    np.testing.assert_allclose(feat, 0.0)


def test_left_hand_only():
    left = _make_dummy_hand(wrist_x=0.25, wrist_y=0.6, wrist_z=-0.05, scale=0.15)
    feat = extract_dynamic_features(left_hand_landmarks=left, right_hand_landmarks=None)

    # Left hand handshape populated, right is zero
    assert not np.all(feat[LEFT_HANDSHAPE_SLICE] == 0.0)
    np.testing.assert_allclose(feat[RIGHT_HANDSHAPE_SLICE], 0.0)

    # Left wrist & scale
    np.testing.assert_allclose(feat[LEFT_WRIST_SLICE], [0.25, 0.6, -0.05], atol=1e-5)
    assert np.isclose(feat[LEFT_SCALE_IDX], 0.15, atol=1e-5)

    # Right wrist & scale must be zero
    np.testing.assert_allclose(feat[RIGHT_WRIST_SLICE], 0.0)
    assert feat[RIGHT_SCALE_IDX] == 0.0

    # Inter-wrist features must be zero because right hand is absent
    np.testing.assert_allclose(feat[INTER_WRIST_VEC_SLICE], 0.0)
    assert feat[INTER_WRIST_DIST_IDX] == 0.0

    # Presence mask
    assert feat[LEFT_PRESENT_IDX] == 1.0
    assert feat[RIGHT_PRESENT_IDX] == 0.0


def test_right_hand_only():
    right = _make_dummy_hand(wrist_x=0.75, wrist_y=0.4, wrist_z=0.02, scale=0.11)
    feat = extract_dynamic_features(left_hand_landmarks=None, right_hand_landmarks=right)

    np.testing.assert_allclose(feat[LEFT_HANDSHAPE_SLICE], 0.0)
    assert not np.all(feat[RIGHT_HANDSHAPE_SLICE] == 0.0)

    np.testing.assert_allclose(feat[LEFT_WRIST_SLICE], 0.0)
    assert feat[LEFT_SCALE_IDX] == 0.0

    np.testing.assert_allclose(feat[RIGHT_WRIST_SLICE], [0.75, 0.4, 0.02], atol=1e-5)
    assert np.isclose(feat[RIGHT_SCALE_IDX], 0.11, atol=1e-5)

    np.testing.assert_allclose(feat[INTER_WRIST_VEC_SLICE], 0.0)
    assert feat[INTER_WRIST_DIST_IDX] == 0.0

    assert feat[LEFT_PRESENT_IDX] == 0.0
    assert feat[RIGHT_PRESENT_IDX] == 1.0


def test_two_hands_spatial_relationship():
    left = _make_dummy_hand(wrist_x=0.2, wrist_y=0.5, wrist_z=0.0, scale=0.1)
    right = _make_dummy_hand(wrist_x=0.8, wrist_y=0.5, wrist_z=0.0, scale=0.1)
    feat = extract_dynamic_features(left, right)

    # Inter-wrist vector: right (0.8, 0.5, 0) - left (0.2, 0.5, 0) = (0.6, 0.0, 0.0)
    expected_vec = np.array([0.6, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(feat[INTER_WRIST_VEC_SLICE], expected_vec, atol=1e-5)
    assert np.isclose(feat[INTER_WRIST_DIST_IDX], 0.6, atol=1e-5)

    assert feat[LEFT_PRESENT_IDX] == 1.0
    assert feat[RIGHT_PRESENT_IDX] == 1.0


def test_controlled_translation_invariance_of_handshape():
    """Verify that moving the hand across the camera modifies wrist position while keeping handshape intact."""
    hand_pos1 = _make_dummy_hand(wrist_x=0.2, wrist_y=0.5, wrist_z=0.0, scale=0.1)
    hand_pos2 = _make_dummy_hand(wrist_x=0.8, wrist_y=0.5, wrist_z=0.0, scale=0.1)

    f1 = extract_dynamic_features(right_hand_landmarks=hand_pos1)
    f2 = extract_dynamic_features(right_hand_landmarks=hand_pos2)

    # Handshape 126-D is unchanged
    np.testing.assert_allclose(f1[HANDSHAPE_SLICE], f2[HANDSHAPE_SLICE], atol=1e-6)

    # But right wrist x position has changed by 0.6!
    assert np.isclose(f1[130], 0.2, atol=1e-5)
    assert np.isclose(f2[130], 0.8, atol=1e-5)


def test_controlled_depth_and_scale_change():
    """Verify that moving closer to camera increases apparent scale."""
    hand_far = _make_dummy_hand(wrist_x=0.5, wrist_y=0.5, wrist_z=0.1, scale=0.08)
    hand_close = _make_dummy_hand(wrist_x=0.5, wrist_y=0.5, wrist_z=-0.1, scale=0.24)

    f_far = extract_dynamic_features(right_hand_landmarks=hand_far)
    f_close = extract_dynamic_features(right_hand_landmarks=hand_close)

    # Handshape 126-D is normalized and invariant
    np.testing.assert_allclose(f_far[HANDSHAPE_SLICE], f_close[HANDSHAPE_SLICE], atol=1e-6)

    # Apparent scale reflects depth: close hand is 3x larger than far hand!
    assert np.isclose(f_far[RIGHT_SCALE_IDX], 0.08, atol=1e-5)
    assert np.isclose(f_close[RIGHT_SCALE_IDX], 0.24, atol=1e-5)
    assert f_close[RIGHT_SCALE_IDX] > f_far[RIGHT_SCALE_IDX]


def test_two_hands_approaching_each_other():
    """Simulates two hands clapping or tapping together (e.g. STOP or MORE)."""
    # Far apart
    l_far = _make_dummy_hand(wrist_x=0.2, wrist_y=0.5)
    r_far = _make_dummy_hand(wrist_x=0.8, wrist_y=0.5)
    f_apart = extract_dynamic_features(l_far, r_far)

    # Touching / close
    l_close = _make_dummy_hand(wrist_x=0.48, wrist_y=0.5)
    r_close = _make_dummy_hand(wrist_x=0.52, wrist_y=0.5)
    f_together = extract_dynamic_features(l_close, r_close)

    # Inter-wrist distance shrinks from 0.60 to 0.04
    assert np.isclose(f_apart[INTER_WRIST_DIST_IDX], 0.60, atol=1e-5)
    assert np.isclose(f_together[INTER_WRIST_DIST_IDX], 0.04, atol=1e-5)
    assert f_apart[INTER_WRIST_DIST_IDX] > f_together[INTER_WRIST_DIST_IDX]
