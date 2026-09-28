import json
import math
from pathlib import Path

names = [
    ("r1", "R = 1.0 (base nominal)", "reports/f5_conv_r1.json"),
    ("q4", "Q = 4.0, R = 1.0", "reports/f5_conv_q4.json"),
    ("r0.25_q4", "R = 0.25, Q = 4.0", "reports/f5_conv_r0.25_q4.json"),
    ("r0.5_q4", "R = 0.5, Q = 4.0", "reports/f5_conv_r0.5_q4.json"),
    ("r0.5_q1", "R = 0.5, Q = 1.0", "reports/f5_conv_r0.5_q1.json"),
    ("r2", "R = 2.0, Q = 1.0", "reports/f5_conv_r2.json"),
    ("r4", "R = 4.0, Q = 1.0", "reports/f5_conv_r4.json"),
    ("q0.25", "Q = 0.25, R = 1.0", "reports/f5_conv_q0.25.json"),
]

print("=" * 80)
print("F5 CONVERGENCE SWEEPS SUMMARY (21 KITTI SEQUENCES)")
print("=" * 80)

for tag, desc, file_name in names:
    p = Path(file_name)
    if not p.exists():
        continue
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    
    pooled = data.get("pooled_diagnostics", {})
    cons = pooled.get("consistency", {})
    c_all = cons.get("all", {})
    ages = cons.get("by_track_age_s", {})
    dist = pooled.get("by_distance_m", {})
    
    # 0-30m pooled
    n_0_30 = sum(dist[k]["n"] for k in ["0-10", "10-20", "20-30"] if k in dist)
    sq_0_30 = sum(dist[k]["n"] * (dist[k]["rmse"]**2) for k in ["0-10", "10-20", "20-30"] if k in dist)
    rmse_0_30 = math.sqrt(sq_0_30 / n_0_30) if n_0_30 > 0 else 0.0
    
    print("-" * 80)
    print(f"RUN: {tag} ({desc}) -> {file_name}")
    print(f"Config: r_scale = {data.get('r_scale')}, q_vehicle = {data.get('q_vehicle')}")
    print(f"  RMSE 0-30m: {rmse_0_30:.3f} m/s | 0-10m: {dist.get('0-10', {}).get('rmse', 0):.3f} | 10-20m: {dist.get('10-20', {}).get('rmse', 0):.3f} | 20-30m: {dist.get('20-30', {}).get('rmse', 0):.3f}")
    print("  Consistency:")
    print(f"    all:       n={c_all.get('n', 0):5d} RMSE V_Z {c_all.get('rmse_vz', 0.0):.2f} - sigma pred {c_all.get('sigma_vz_pred_rms', 0.0):.2f} (ratio {c_all.get('ratio', 0.0):.2f}) - NIS mean {c_all.get('mean_nis', 0.0):.2f} (>5.99: {c_all.get('frac_nis_gt_5.99', 0.0)*100:.0f} %)")
    for age_key in ["1_2", "2_4", "4_inf"]:
        c_age = ages.get(age_key, {})
        label = f"age {age_key} s"
        print(f"    {label:>10s}: n={c_age.get('n', 0):5d} RMSE V_Z {c_age.get('rmse_vz', 0.0):.2f} - sigma pred {c_age.get('sigma_vz_pred_rms', 0.0):.2f} (ratio {c_age.get('ratio', 0.0):.2f}) - NIS mean {c_age.get('mean_nis', 0.0):.2f} (>5.99: {c_age.get('frac_nis_gt_5.99', 0.0)*100:.0f} %)")
    print()
