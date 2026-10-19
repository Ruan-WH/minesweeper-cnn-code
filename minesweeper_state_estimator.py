import numpy as np
import torch
from tqdm import tqdm
from collections import Counter, defaultdict

from minesweeper_pygame import MinesweeperEnv
from constants import BrickState
from lib import models
from lib.wrappers import CustomObservationWrapper


def get_valid_random_action(state):
    """Return a valid random action (epsilon exploration)."""
    flat_board = state[-1].reshape(-1)  # unknown mask
    unknown_indices = np.where(flat_board == 1)[0]
    if len(unknown_indices) == 0:
        return None
    return int(np.random.choice(unknown_indices))


def get_cnn_action_with_exploration(state, model, device, epsilon=0.05):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        action = get_valid_random_action(state)
        if action is not None:
            return action

    with torch.no_grad():
        input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probability_array = model(input_tensor).detach().cpu().numpy().squeeze()

    mask = state[-1].reshape(-1) == 0
    probability_array[mask] = np.inf

    min_index = int(np.argmin(probability_array))
    return None if probability_array[min_index] == np.inf else min_index


def _safe_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _safe_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_state, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return next_state, reward, done, info
    return out


def _depth_bin(revealed_ratio, n_bins=6):
    rr = min(max(revealed_ratio, 0.10), 0.999999)
    x = (rr - 0.10) / 0.90
    b = int(np.floor(x * n_bins))
    return max(0, min(n_bins - 1, b))


def _canonical_board_bytes(player_board):
    """Canonicalize board by 8 dihedral symmetries."""
    transforms = []
    for k in range(4):
        r = np.rot90(player_board, k)
        transforms.append(r)
        transforms.append(np.fliplr(r))
    return min(arr.tobytes() for arr in transforms)


def _richness_metrics(counter, sample_count):
    """Compute richness estimators from one counter."""
    S_obs = len(counter)
    if sample_count <= 0 or S_obs == 0:
        return {
            "S_obs": 0.0,
            "f1": 0,
            "f2": 0,
            "f3": 0,
            "f4": 0,
            "coverage": 0.0,
            "unseen_mass": 1.0,
            "chao1": 0.0,
            "ichao1": 0.0,
            "jack1": 0.0,
            "jack2": 0.0,
            "bootstrap": 0.0,
            "robust": 0.0,
            "lower": 0.0,
            "upper": 0.0,
        }

    counts = np.fromiter(counter.values(), dtype=np.int64)
    f1 = int(np.sum(counts == 1))
    f2 = int(np.sum(counts == 2))
    f3 = int(np.sum(counts == 3))
    f4 = int(np.sum(counts == 4))

    coverage = max(0.0, 1.0 - (f1 / sample_count))
    unseen_mass = 1.0 - coverage

    # Bias-corrected Chao1 as lower-bound style estimator in heavy-tail settings.
    chao1 = S_obs + (f1 * (f1 - 1)) / (2.0 * (f2 + 1))

    denom = 4.0 * f4 if f4 > 0 else 4.0
    inner = f1 - (f2 * f3) / (2.0 * (f4 + 1))
    ichao1_extra = (f3 / denom) * inner if inner > 0 else 0.0
    ichao1 = chao1 + ichao1_extra

    jack1 = S_obs + f1 * (sample_count - 1) / sample_count
    if sample_count > 1:
        jack2 = (
            S_obs
            + f1 * (2.0 * sample_count - 3.0) / sample_count
            - f2 * ((sample_count - 2.0) ** 2) / (sample_count * (sample_count - 1.0))
        )
    else:
        jack2 = jack1

    p = counts / sample_count
    bootstrap = S_obs + float(np.sum((1.0 - p) ** sample_count))

    candidates = np.array([chao1, ichao1, jack1, jack2, bootstrap], dtype=np.float64)
    robust = float(np.median(candidates))

    return {
        "S_obs": float(S_obs),
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "coverage": float(coverage),
        "unseen_mass": float(unseen_mass),
        "chao1": float(chao1),
        "ichao1": float(ichao1),
        "jack1": float(jack1),
        "jack2": float(jack2),
        "bootstrap": float(bootstrap),
        "robust": robust,
        "lower": float(np.min(candidates)),
        "upper": float(np.max(candidates)),
    }


def _fit_saturation_upper_bound(samples, lower_series, z=1.96):
    """
    Fit S(N) = S_inf * N / (N + B) on the lower-bound series and return
    (S_inf_point, S_inf_upper_95, B, R2).

    Linearized form: N/S = a*N + b, where a = 1/S_inf, b = B/S_inf.
    """
    if len(samples) < 4:
        return None

    x = np.asarray(samples, dtype=np.float64)
    s = np.asarray(lower_series, dtype=np.float64)
    valid = (x > 0) & (s > 0)
    x = x[valid]
    s = s[valid]

    if len(x) < 4:
        return None

    y = x / s
    X = np.column_stack([x, np.ones_like(x)])

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a_hat, b_hat = float(beta[0]), float(beta[1])
    except np.linalg.LinAlgError:
        return None

    if not np.isfinite(a_hat) or not np.isfinite(b_hat) or a_hat <= 0:
        return None

    y_hat = X @ beta
    resid = y - y_hat
    n = len(y)
    if n <= 2:
        return None

    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1.0 - rss / tss

    dof = n - 2
    sigma2 = rss / max(dof, 1)

    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None

    var_a = float(sigma2 * xtx_inv[0, 0])
    se_a = np.sqrt(max(var_a, 0.0))

    # Upper bound for S_inf comes from lower confidence bound of a.
    a_low = max(a_hat - z * se_a, 1e-15)
    s_inf_point = 1.0 / a_hat
    s_inf_upper = 1.0 / a_low
    b_point = max(0.0, b_hat * s_inf_point)

    if not (np.isfinite(s_inf_point) and np.isfinite(s_inf_upper) and np.isfinite(b_point)):
        return None

    return {
        "S_inf_point": float(s_inf_point),
        "S_inf_upper": float(s_inf_upper),
        "B_point": float(b_point),
        "r2": float(r2),
    }


def run_state_count_estimation(
    env_config,
    model_path,
    total_episodes=150000,
    eval_interval=10000,
    epsilon=0.05,
    min_reveal_ratio=0.10,
    depth_bins=6,
    use_symmetry_canonicalization=True,
):
    r, c, m = env_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_env = MinesweeperEnv(num_rows=r, num_cols=c, num_mines=m, render_mode="rgb_array")
    env = CustomObservationWrapper(raw_env, encoding_type="one_hot")

    model = models.Model(env.observation_space.shape, env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\n--- Starting Interval Estimation for {r}x{c} (Mines: {m}) ---")
    print(f"Exploration epsilon: {epsilon}")
    print(f"Min reveal ratio: {min_reveal_ratio}")
    print(f"Depth bins: {depth_bins}")
    print(f"Symmetry canonicalization: {use_symmetry_canonicalization}")

    global_counter = Counter()
    global_samples = 0

    bin_counters = defaultdict(Counter)
    bin_samples = Counter()

    total_cells = env.num_rows * env.num_cols

    history = []

    for episode in tqdm(range(1, total_episodes + 1), desc="Sampling"):
        state = _safe_reset(env)
        env.button = "left"
        done = False

        while not done:
            action = get_cnn_action_with_exploration(state, model, device, epsilon)
            if action is None:
                break

            next_state, _, done, _ = _safe_step(env, action)
            state = next_state

            player_board = env.player_board
            revealed_cells = np.count_nonzero(
                (player_board != BrickState.UNKNOWN) & (player_board != BrickState.FLAG)
            )
            revealed_ratio = revealed_cells / total_cells

            if revealed_ratio >= min_reveal_ratio:
                if use_symmetry_canonicalization:
                    state_hash = _canonical_board_bytes(player_board)
                else:
                    state_hash = player_board.tobytes()

                global_counter[state_hash] += 1
                global_samples += 1

                b = _depth_bin(revealed_ratio, n_bins=depth_bins)
                bin_counters[b][state_hash] += 1
                bin_samples[b] += 1

        if episode % eval_interval == 0 or episode == total_episodes:
            global_metrics = _richness_metrics(global_counter, global_samples)

            agg_obs = 0.0
            agg_robust = 0.0
            agg_lower = 0.0
            agg_upper = 0.0
            weighted_unseen_mass_num = 0.0
            weighted_unseen_mass_den = 0.0

            for b in range(depth_bins):
                mtr = _richness_metrics(bin_counters[b], int(bin_samples[b]))
                agg_obs += mtr["S_obs"]
                agg_robust += mtr["robust"]
                agg_lower += mtr["lower"]
                agg_upper += mtr["upper"]
                weighted_unseen_mass_num += mtr["unseen_mass"] * bin_samples[b]
                weighted_unseen_mass_den += bin_samples[b]

            agg_unseen_mass = (
                weighted_unseen_mass_num / weighted_unseen_mass_den
                if weighted_unseen_mass_den > 0
                else 1.0
            )

            # Use global Chao1 as the lower bound series for interval fitting.
            lower_bound = max(global_metrics["chao1"], global_metrics["S_obs"])

            history.append(
                {
                    "episode": episode,
                    "samples": global_samples,
                    "lower": lower_bound,
                    "agg_obs": agg_obs,
                    "agg_robust": agg_robust,
                    "agg_unseen_mass": agg_unseen_mass,
                }
            )

            sat = _fit_saturation_upper_bound(
                [h["samples"] for h in history],
                [h["lower"] for h in history],
                z=1.96,
            )

            if sat is not None:
                upper_bound = max(lower_bound, sat["S_inf_upper"])
            else:
                upper_bound = np.nan

            growth = np.nan
            if len(history) >= 2:
                prev = history[-2]
                growth = (history[-1]["lower"] - prev["lower"]) / max(prev["lower"], 1.0)

            print(f"\n[Episode {episode}]")
            print(f"  Global samples: {global_samples:,}")
            print(
                f"  Global S_obs: {int(global_metrics['S_obs']):,} | "
                f"Global Chao1(lower): {int(global_metrics['chao1']):,} | "
                f"Coverage: {global_metrics['coverage']:.4f}"
            )
            print(f"  Stratified S_obs(sum bins): {int(agg_obs):,}")
            print(
                f"  Stratified Robust(ref): {agg_robust:.3e} "
                f"(approx. {int(agg_robust):,})"
            )
            print(
                f"  Stratified Range(ref): [{int(agg_lower):,}, {int(agg_upper):,}]"
            )
            print(f"  Stratified unseen-mass(ref): {agg_unseen_mass:.6f}")
            if np.isfinite(growth):
                print(f"  Lower-bound relative growth since last eval: {growth:.6%}")

            if sat is not None:
                print(
                    f"  Saturation fit on lower-bound curve: "
                    f"S_inf(point)={int(sat['S_inf_point']):,}, "
                    f"S_inf(upper95)={int(sat['S_inf_upper']):,}, "
                    f"R^2={sat['r2']:.4f}"
                )
                print(
                    f"  >>> Interval Estimate: [{int(lower_bound):,}, {int(upper_bound):,}] <<<"
                )
            else:
                print(
                    f"  >>> Interval Estimate: [{int(lower_bound):,}, N/A] "
                    f"(need >=4 checkpoints for upper bound fit) <<<"
                )

    env.close()


if __name__ == "__main__":
    target_config = (9, 9, 10)
    model_file_path = r"./models/CNN-1763950674/Minesweeper-v0-best_win_rate.dat"

    run_state_count_estimation(
        target_config,
        model_file_path,
        total_episodes=150000,
        eval_interval=10000,
        epsilon=0.05,
        min_reveal_ratio=0.10,
        depth_bins=6,
        use_symmetry_canonicalization=True,
    )
