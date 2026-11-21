import sys
sys.path.insert(0, '.')
import cv2
import json
import numpy as np
from pathlib import Path
from scipy import stats

renders_dir = Path('renders/phase5_5')
gt_dir = Path('data/val_future_videos')
scenarios = ['0000900', '0000901', '0000902', '0000903', '0000904', '0000905', '0000906', '0000911', '0000919', '0000984']
candidates = ['vanilla_svd', 'project_baseline', 'cssc_only']
cameras = ['front', 'front_left', 'front_right', 'rear', 'rear_left', 'rear_right']

from scripts.run_candidate_head_to_head import compute_true_psnr, compute_video_ssim

def load_video(p):
    cap = cv2.VideoCapture(str(p))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)

# Load existing 9-scenario metrics if available
cached_data = {}
cache_file = Path('reports/phase5_5_candidate_head_to_head.json')
if cache_file.exists():
    with open(cache_file) as f:
        prev_rep = json.load(f)
        for row in prev_rep.get('predictions', []):
            cached_data[(row['scenario_id'], row['camera'], row['candidate'])] = (row['psnr'], row['ssim'])

# 1. Full 180 matrix audit
matrix = []
missing = []
present = []

for sid in scenarios:
    for cand in candidates:
        for cam in cameras:
            fname = f'{sid}_{cam}_{cand}.mp4'
            fpath = renders_dir / fname
            if fpath.exists():
                size = fpath.stat().st_size
                present.append((sid, cand, cam, fname, size))
                matrix.append({'scenario': sid, 'candidate': cand, 'camera': cam, 'file': fname, 'status': 'PRESENT', 'bytes': size})
            else:
                missing.append((sid, cand, cam, fname))
                matrix.append({'scenario': sid, 'candidate': cand, 'camera': cam, 'file': fname, 'status': 'MISSING', 'bytes': 0})

print(f'Total expected: {len(matrix)}, Present: {len(present)}, Missing: {len(missing)}')
for m in missing:
    print(f'MISSING FILE: {m}')

# 2. Compute paired scenario-level differences
per_scenario_data = {}
for sid in scenarios:
    per_scenario_data[sid] = {}
    common_cams = [cam for cam in cameras if all((renders_dir / f'{sid}_{cam}_{cand}.mp4').exists() for cand in candidates)]
    for cand in candidates:
        psnr_common, ssim_common = [], []
        psnr_all, ssim_all = [], []
        for cam in cameras:
            vpath = renders_dir / f'{sid}_{cam}_{cand}.mp4'
            gtpath = gt_dir / sid / f'{cam}.mp4'
            if vpath.exists() and gtpath.exists():
                if (sid, cam, cand) in cached_data:
                    p, s = cached_data[(sid, cam, cand)]
                else:
                    pred = load_video(vpath)
                    gt = load_video(gtpath)
                    p = compute_true_psnr(pred, gt)
                    s = compute_video_ssim(pred, gt)
                psnr_all.append(p)
                ssim_all.append(s)
                if cam in common_cams:
                    psnr_common.append(p)
                    ssim_common.append(s)
        per_scenario_data[sid][cand] = {
            'mean_psnr_common': float(np.mean(psnr_common)),
            'mean_ssim_common': float(np.mean(ssim_common)),
            'mean_psnr_all': float(np.mean(psnr_all)),
            'mean_ssim_all': float(np.mean(ssim_all)),
            'n_common': len(common_cams),
            'n_all': len(psnr_all),
            'psnr_common_list': [float(x) for x in psnr_common],
            'ssim_common_list': [float(x) for x in ssim_common],
        }

# Aggregate across all 10 scenarios (common camera view)
baseline_psnr = [per_scenario_data[s]['project_baseline']['mean_psnr_common'] for s in scenarios]
vanilla_psnr  = [per_scenario_data[s]['vanilla_svd']['mean_psnr_common'] for s in scenarios]
cssc_psnr     = [per_scenario_data[s]['cssc_only']['mean_psnr_common'] for s in scenarios]

baseline_ssim = [per_scenario_data[s]['project_baseline']['mean_ssim_common'] for s in scenarios]
vanilla_ssim  = [per_scenario_data[s]['vanilla_svd']['mean_ssim_common'] for s in scenarios]
cssc_ssim     = [per_scenario_data[s]['cssc_only']['mean_ssim_common'] for s in scenarios]

diff_base_van_psnr = np.array(baseline_psnr) - np.array(vanilla_psnr)
diff_base_cssc_psnr = np.array(baseline_psnr) - np.array(cssc_psnr)
diff_base_van_ssim = np.array(baseline_ssim) - np.array(vanilla_ssim)
diff_base_cssc_ssim = np.array(baseline_ssim) - np.array(cssc_ssim)

# Bootstrap 95% CIs (10,000 resamples)
rng = np.random.default_rng(42)
def bootstrap_ci(diffs, n_resamples=10000):
    resampled_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_resamples)]
    return np.percentile(resampled_means, 2.5), np.percentile(resampled_means, 97.5)

ci_base_van_psnr = bootstrap_ci(diff_base_van_psnr)
ci_base_cssc_psnr = bootstrap_ci(diff_base_cssc_psnr)
ci_base_van_ssim = bootstrap_ci(diff_base_van_ssim)
ci_base_cssc_ssim = bootstrap_ci(diff_base_cssc_ssim)

# Paired t-test
t_base_van_p = float(stats.ttest_rel(baseline_psnr, vanilla_psnr).pvalue)
t_base_cssc_p = float(stats.ttest_rel(baseline_psnr, cssc_psnr).pvalue)

print('\n=== 10-SCENARIO COMMON-CAMERA AGGREGATE SUMMARY ===')
print(f'Project Baseline: Mean PSNR={np.mean(baseline_psnr):.3f} +/- {np.std(baseline_psnr):.3f}, Mean SSIM={np.mean(baseline_ssim):.4f} +/- {np.std(baseline_ssim):.4f}')
print(f'CSSC-Only:        Mean PSNR={np.mean(cssc_psnr):.3f} +/- {np.std(cssc_psnr):.3f}, Mean SSIM={np.mean(cssc_ssim):.4f} +/- {np.std(cssc_ssim):.4f}')
print(f'Vanilla SVD:      Mean PSNR={np.mean(vanilla_psnr):.3f} +/- {np.std(vanilla_psnr):.3f}, Mean SSIM={np.mean(vanilla_ssim):.4f} +/- {np.std(vanilla_ssim):.4f}')

print('\n=== PAIRED DIFFERENCES & 95% BOOTSTRAP CI ===')
print(f'Delta PSNR (Baseline - Vanilla): mean={np.mean(diff_base_van_psnr):+.3f} dB, 95% CI=[{ci_base_van_psnr[0]:+.3f}, {ci_base_van_psnr[1]:+.3f}], p={t_base_van_p:.4f}')
print(f'Delta PSNR (Baseline - CSSC):    mean={np.mean(diff_base_cssc_psnr):+.3f} dB, 95% CI=[{ci_base_cssc_psnr[0]:+.3f}, {ci_base_cssc_psnr[1]:+.3f}], p={t_base_cssc_p:.4f}')
print(f'Delta SSIM (Baseline - Vanilla): mean={np.mean(diff_base_van_ssim):+.4f}, 95% CI=[{ci_base_van_ssim[0]:+.4f}, {ci_base_van_ssim[1]:+.4f}]')
print(f'Delta SSIM (Baseline - CSSC):    mean={np.mean(diff_base_cssc_ssim):+.4f}, 95% CI=[{ci_base_cssc_ssim[0]:+.4f}, {ci_base_cssc_ssim[1]:+.4f}]')

win_psnr_vs_cssc = int(sum(d > 0 for d in diff_base_cssc_psnr))
win_psnr_vs_van = int(sum(d > 0 for d in diff_base_van_psnr))
print(f'\nScenario Win Rates (PSNR): Baseline vs CSSC = {win_psnr_vs_cssc}/10 ({win_psnr_vs_cssc*10}%), Baseline vs Vanilla = {win_psnr_vs_van}/10 ({win_psnr_vs_van*10}%)')

stats_payload = {
    'total_expected': 180,
    'total_present': len(present),
    'total_missing': len(missing),
    'missing_files': [m[3] for m in missing],
    'audit_matrix': matrix,
    'per_scenario': per_scenario_data,
    'aggregate_common_camera': {
        'project_baseline': {'mean_psnr': float(np.mean(baseline_psnr)), 'std_psnr': float(np.std(baseline_psnr)), 'mean_ssim': float(np.mean(baseline_ssim)), 'std_ssim': float(np.std(baseline_ssim))},
        'cssc_only': {'mean_psnr': float(np.mean(cssc_psnr)), 'std_psnr': float(np.std(cssc_psnr)), 'mean_ssim': float(np.mean(cssc_ssim)), 'std_ssim': float(np.std(cssc_ssim))},
        'vanilla_svd': {'mean_psnr': float(np.mean(vanilla_psnr)), 'std_psnr': float(np.std(vanilla_psnr)), 'mean_ssim': float(np.mean(vanilla_ssim)), 'std_ssim': float(np.std(vanilla_ssim))},
    },
    'paired_differences': {
        'baseline_minus_vanilla': {
            'mean_delta_psnr': float(np.mean(diff_base_van_psnr)),
            'ci_95_delta_psnr': [float(ci_base_van_psnr[0]), float(ci_base_van_psnr[1])],
            'mean_delta_ssim': float(np.mean(diff_base_van_ssim)),
            'ci_95_delta_ssim': [float(ci_base_van_ssim[0]), float(ci_base_van_ssim[1])],
            'win_rate_psnr': f'{win_psnr_vs_van}/10',
            'p_value_psnr': t_base_van_p
        },
        'baseline_minus_cssc': {
            'mean_delta_psnr': float(np.mean(diff_base_cssc_psnr)),
            'ci_95_delta_psnr': [float(ci_base_cssc_psnr[0]), float(ci_base_cssc_psnr[1])],
            'mean_delta_ssim': float(np.mean(diff_base_cssc_ssim)),
            'ci_95_delta_ssim': [float(ci_base_cssc_ssim[0]), float(ci_base_cssc_ssim[1])],
            'win_rate_psnr': f'{win_psnr_vs_cssc}/10',
            'p_value_psnr': t_base_cssc_p
        }
    }
}

Path('reports').mkdir(parents=True, exist_ok=True)
with open('reports/phase5_5_2_final_audit_and_freeze.json', 'w') as f:
    json.dump(stats_payload, f, indent=2)
print('Successfully saved reports/phase5_5_2_final_audit_and_freeze.json')
