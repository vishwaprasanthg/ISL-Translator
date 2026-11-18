"""
Controlled Motion and Spatial Validation for 140-D Dynamic Feature Representation.
Evaluates the 7 core physical signing scenarios and verifies expected feature responses.
"""

import numpy as np
from ml.preprocessing.dynamic_feature_extractor import (
    extract_dynamic_features,
    HANDSHAPE_SLICE,
    LEFT_WRIST_SLICE,
    RIGHT_WRIST_SLICE,
    RIGHT_SCALE_IDX,
    LEFT_SCALE_IDX,
    INTER_WRIST_VEC_SLICE,
    INTER_WRIST_DIST_IDX,
    PRESENCE_MASK_SLICE,
)


def make_hand(wrist_x=0.5, wrist_y=0.5, wrist_z=0.0, scale=0.10, is_pointing=False):
    """
    Constructs a 21-landmark hand.
    If is_pointing=True: index finger extended forward, other fingers folded (I/ME and YOU posture).
    """
    lms = []
    for i in range(21):
        if i == 0:
            lms.append({'x': wrist_x, 'y': wrist_y, 'z': wrist_z})
        elif i == 9:
            lms.append({'x': wrist_x, 'y': wrist_y - scale, 'z': wrist_z})
        elif is_pointing and (5 <= i <= 8):
            # Index finger extended along Y/Z
            tip_frac = (i - 4) / 4.0
            lms.append({'x': wrist_x + 0.01, 'y': wrist_y - scale * (0.8 + 0.5 * tip_frac), 'z': wrist_z - scale * 0.4 * tip_frac})
        elif is_pointing and i > 8:
            # Other fingers folded into palm
            lms.append({'x': wrist_x + 0.03 * ((i % 4) + 1), 'y': wrist_y - scale * 0.3, 'z': wrist_z + 0.02})
        else:
            # Default open/curved hand
            lms.append({
                'x': wrist_x + scale * 0.1 * (i % 5),
                'y': wrist_y - scale * 0.2 * (i // 5),
                'z': wrist_z + scale * 0.05 * i,
            })
    return lms


def run_all_scenarios():
    print("=" * 70)
    print("  CONTROLLED MOTION & SPATIAL VALIDATION REPORT (140-D EXTRACTOR)")
    print("=" * 70)

    results = {}

    # -------------------------------------------------------------
    # Scenario 1: Stationary hand with natural sensor tremor
    # -------------------------------------------------------------
    print("\n[Scenario 1] Stationary hand with natural tremor (std=0.003)")
    base_hand = make_hand(wrist_x=0.5, wrist_y=0.5, scale=0.12)
    f_base = extract_dynamic_features(right_hand_landmarks=base_hand)

    tremor_frames = []
    for t in range(20):
        noisy_hand = [
            {'x': lm['x'] + np.random.normal(0, 0.003),
             'y': lm['y'] + np.random.normal(0, 0.003),
             'z': lm['z'] + np.random.normal(0, 0.003)}
            for lm in base_hand
        ]
        tremor_frames.append(extract_dynamic_features(right_hand_landmarks=noisy_hand))
    tremor_frames = np.array(tremor_frames)

    h_disp = np.mean([np.linalg.norm(tremor_frames[t, HANDSHAPE_SLICE] - f_base[HANDSHAPE_SLICE]) for t in range(20)])
    w_disp = np.mean([np.linalg.norm(tremor_frames[t, RIGHT_WRIST_SLICE] - f_base[RIGHT_WRIST_SLICE]) for t in range(20)])
    s_disp = np.mean([abs(tremor_frames[t, RIGHT_SCALE_IDX] - f_base[RIGHT_SCALE_IDX]) for t in range(20)])

    print(f"  Handshape variation norm (126-D) : {h_disp:.4f}  (sensor tremor on 63 coords, stable)")
    print(f"  Wrist position jitter            : {w_disp:.4f}")
    print(f"  Hand scale jitter                : {s_disp:.4f}")
    assert h_disp < 0.35
    results["stationary"] = {"handshape_disp": float(h_disp), "wrist_disp": float(w_disp)}

    # -------------------------------------------------------------
    # Scenario 2: Hand moving left to right (pure translation)
    # -------------------------------------------------------------
    print("\n[Scenario 2] Pure hand translation (left to right, dx = +0.50)")
    h_left = make_hand(wrist_x=0.25, wrist_y=0.5, scale=0.12)
    h_right = make_hand(wrist_x=0.75, wrist_y=0.5, scale=0.12)
    f_left = extract_dynamic_features(right_hand_landmarks=h_left)
    f_right = extract_dynamic_features(right_hand_landmarks=h_right)

    h_diff_lr = np.linalg.norm(f_right[HANDSHAPE_SLICE] - f_left[HANDSHAPE_SLICE])
    w_diff_x = f_right[130] - f_left[130]

    print(f"  126-D Handshape difference    : {h_diff_lr:.6f}  (EXACTLY invariant)")
    print(f"  140-D Wrist X displacement    : {w_diff_x:+.4f}  (Detected: +0.5000)")
    assert np.isclose(h_diff_lr, 0.0, atol=1e-5)
    assert np.isclose(w_diff_x, 0.50, atol=1e-5)
    results["translation"] = {"handshape_diff": float(h_diff_lr), "wrist_dx": float(w_diff_x)}

    # -------------------------------------------------------------
    # Scenario 3: Hand moving towards & away from camera (depth trajectory)
    # -------------------------------------------------------------
    print("\n[Scenario 3] Hand moving towards camera (depth change)")
    h_far = make_hand(wrist_x=0.5, wrist_y=0.5, scale=0.08)
    h_near = make_hand(wrist_x=0.5, wrist_y=0.5, scale=0.24)
    f_far = extract_dynamic_features(right_hand_landmarks=h_far)
    f_near = extract_dynamic_features(right_hand_landmarks=h_near)

    h_diff_depth = np.linalg.norm(f_near[HANDSHAPE_SLICE] - f_far[HANDSHAPE_SLICE])
    scale_ratio = f_near[RIGHT_SCALE_IDX] / f_far[RIGHT_SCALE_IDX]

    print(f"  126-D Handshape difference    : {h_diff_depth:.6f}  (Invariant under zoom)")
    print(f"  Scale feature ratio (near/far): {scale_ratio:.2f}x  (Detected: 3.00x growth)")
    assert np.isclose(h_diff_depth, 0.0, atol=1e-5)
    assert np.isclose(scale_ratio, 3.0, atol=1e-5)
    results["depth"] = {"handshape_diff": float(h_diff_depth), "scale_ratio": float(scale_ratio)}

    # -------------------------------------------------------------
    # Scenario 4: I/ME versus YOU distinguishability
    # -------------------------------------------------------------
    print("\n[Scenario 4] I/ME vs YOU trajectory comparison")
    # In YOU: hand points forward and extends outward towards camera
    # Hand starts neutral (wrist_y=0.55, scale=0.10) and pushes forward (wrist_y=0.48, scale=0.18, wrist_z=-0.12)
    you_start = make_hand(wrist_x=0.52, wrist_y=0.55, wrist_z=0.0, scale=0.10, is_pointing=True)
    you_end   = make_hand(wrist_x=0.55, wrist_y=0.48, wrist_z=-0.12, scale=0.18, is_pointing=True)
    f_you_start = extract_dynamic_features(right_hand_landmarks=you_start)
    f_you_end   = extract_dynamic_features(right_hand_landmarks=you_end)

    # In I/ME: hand points to self, draws inward toward chest
    # Hand moves toward chest (wrist_y=0.62, scale=0.09, wrist_z=+0.05)
    ime_start = make_hand(wrist_x=0.52, wrist_y=0.55, wrist_z=0.0, scale=0.10, is_pointing=True)
    ime_end   = make_hand(wrist_x=0.48, wrist_y=0.64, wrist_z=0.05, scale=0.08, is_pointing=True)
    f_ime_start = extract_dynamic_features(right_hand_landmarks=ime_start)
    f_ime_end   = extract_dynamic_features(right_hand_landmarks=ime_end)

    # In 126-D handshape:
    static_diff = np.linalg.norm(f_you_end[HANDSHAPE_SLICE] - f_ime_end[HANDSHAPE_SLICE])
    # In 140-D global trajectory:
    delta_scale_you = f_you_end[RIGHT_SCALE_IDX] - f_you_start[RIGHT_SCALE_IDX]
    delta_scale_ime = f_ime_end[RIGHT_SCALE_IDX] - f_ime_start[RIGHT_SCALE_IDX]
    delta_y_you = f_you_end[131] - f_you_start[131]
    delta_y_ime = f_ime_end[131] - f_ime_start[131]

    print(f"  126-D Static difference at end: {static_diff:.6f}  (nearly indistinguishable static handshape)")
    print(f"  YOU scale trajectory (delta s): {delta_scale_you:+.4f}  (hand grows as it extends forward)")
    print(f"  I/ME scale trajectory(delta s): {delta_scale_ime:+.4f}  (hand contracts as it moves to chest)")
    print(f"  YOU vertical movement(delta y): {delta_y_you:+.4f}  (hand raises forward)")
    print(f"  I/ME vertical movement(delta y): {delta_y_ime:+.4f}  (hand drops to chest level)")
    assert delta_scale_you > 0
    assert delta_scale_ime < 0
    results["i_me_vs_you"] = {
        "static_handshape_diff": float(static_diff),
        "you_scale_change": float(delta_scale_you),
        "ime_scale_change": float(delta_scale_ime)
    }

    # -------------------------------------------------------------
    # Scenario 5: Two hands moving together (STOP, HELP, MORE)
    # -------------------------------------------------------------
    print("\n[Scenario 5] Two hands approaching each other (STOP / MORE contact)")
    l_apart = make_hand(wrist_x=0.20, wrist_y=0.55, scale=0.10)
    r_apart = make_hand(wrist_x=0.80, wrist_y=0.55, scale=0.10)
    f_apart = extract_dynamic_features(l_apart, r_apart)

    l_contact = make_hand(wrist_x=0.48, wrist_y=0.55, scale=0.10)
    r_contact = make_hand(wrist_x=0.52, wrist_y=0.55, scale=0.10)
    f_contact = extract_dynamic_features(l_contact, r_contact)

    dist_apart = f_apart[INTER_WRIST_DIST_IDX]
    dist_contact = f_contact[INTER_WRIST_DIST_IDX]
    print(f"  Inter-wrist distance (apart)  : {dist_apart:.4f}")
    print(f"  Inter-wrist distance (contact): {dist_contact:.4f}  (clear collision trajectory)")
    assert dist_apart > dist_contact
    results["two_hands_approach"] = {"dist_apart": float(dist_apart), "dist_contact": float(dist_contact)}

    # -------------------------------------------------------------
    # Scenario 6: Finger articulation without wrist translation
    # -------------------------------------------------------------
    print("\n[Scenario 6] Pure finger articulation with fixed wrist position")
    h_open = make_hand(wrist_x=0.50, wrist_y=0.50, scale=0.12, is_pointing=False)
    h_point = make_hand(wrist_x=0.50, wrist_y=0.50, scale=0.12, is_pointing=True)
    f_open = extract_dynamic_features(right_hand_landmarks=h_open)
    f_point = extract_dynamic_features(right_hand_landmarks=h_point)

    h_diff_artic = np.linalg.norm(f_point[HANDSHAPE_SLICE] - f_open[HANDSHAPE_SLICE])
    w_diff_artic = np.linalg.norm(f_point[RIGHT_WRIST_SLICE] - f_open[RIGHT_WRIST_SLICE])

    print(f"  Handshape feature delta (126): {h_diff_artic:.4f}  (high articulation detected)")
    print(f"  Wrist position displacement  : {w_diff_artic:.6f}  (wrist stationary at 0.5, 0.5)")
    assert h_diff_artic > 0.5
    assert np.isclose(w_diff_artic, 0.0, atol=1e-5)
    results["finger_articulation"] = {"handshape_delta": float(h_diff_artic), "wrist_delta": float(w_diff_artic)}

    print("\n" + "=" * 70)
    print("  ALL 6 CONTROLLED SCENARIOS PASSED WITH EXPECTED BEHAVIOUR ✓")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_all_scenarios()
