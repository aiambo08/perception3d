import json
from pathlib import Path

names = [
    ("r1", "R = 1.0 (base)", "reports/f5_conv_r1.json"),
    ("r2", "R = 2.0", "reports/f5_conv_r2.json"),
    ("r4", "R = 4.0", "reports/f5_conv_r4.json"),
    ("q0.25", "Q = 0.25", "reports/f5_conv_q0.25.json"),
    ("q4", "Q = 4.0", "reports/f5_conv_q4.json"),
]

for tag, desc, file_name in names:
    p = Path(file_name)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    
    pooled = data.get("pooled_diagnostics", {})
    cons = pooled.get("consistency", {})
    c_all = cons.get("all", {})
    ages = cons.get("by_track_age_s", {})
    
    print("=" * 80)
    print(f"RUN: {tag} ({desc}) -> {file_name}")
    print(f"Config: r_scale = {data.get('r_scale')}, q_vehicle = {data.get('q_vehicle')}")
    print("=" * 80)
    print("pooled (21 seqs):")
    print(f"    consistency         all: n={c_all.get('n', 0):5d} RMSE V_Z {c_all.get('rmse_vz', 0.0):.2f} · σ pred {c_all.get('sigma_vz_pred_rms', 0.0):.2f} (ratio {c_all.get('ratio', 0.0):.2f}) · NIS mean {c_all.get('mean_nis', 0.0):.2f} (>5.99: {c_all.get('frac_nis_gt_5.99', 0.0)*100:.0f} %)")
    for age_key in ["1_2", "2_4", "4_inf"]:
        c_age = ages.get(age_key, {})
        label = f"age {age_key} s"
        print(f"    consistency {label:>11s}: n={c_age.get('n', 0):5d} RMSE V_Z {c_age.get('rmse_vz', 0.0):.2f} · σ pred {c_age.get('sigma_vz_pred_rms', 0.0):.2f} (ratio {c_age.get('ratio', 0.0):.2f}) · NIS mean {c_age.get('mean_nis', 0.0):.2f} (>5.99: {c_age.get('frac_nis_gt_5.99', 0.0)*100:.0f} %)")
    print(f"wrote {file_name}\n")
