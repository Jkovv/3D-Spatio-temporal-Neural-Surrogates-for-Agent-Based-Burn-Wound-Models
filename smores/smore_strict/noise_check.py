"""How large is the stochastic spread of the ABM between replicates, compared
with the spread between sweep points? Uses only points with all realisations."""
import numpy as np, sys
D = np.load(sys.argv[1]); Y = D["Y"]; cyts = [str(c) for c in D["cyts"]]
full = np.isfinite(Y[:, :, 0, 0]).all(1); Y = Y[full]; R = Y.shape[1]
print(f"points with all {R} realisations: {Y.shape[0]}\n")
print(f"{'cyt':5s} {'replicate CV':>13s} {'SE of mean (rel)':>17s} {'between-point CV':>17s} {'signal/noise':>13s}")
for c, name in enumerate(cyts):
    y = Y[:, :, 20:, c]                                  # skip the first 20 h, where concentrations start near zero
    mu = y.mean(1); sd = y.std(1, ddof=1)
    cv = np.nanmedian(sd / np.where(np.abs(mu) > 0, np.abs(mu), np.nan))
    se = cv / np.sqrt(R)
    btw = np.nanmedian(mu.std(0) / np.abs(mu.mean(0)))
    print(f"{name:5s} {cv:13.3f} {se:17.3f} {btw:17.3f} {btw/cv:13.1f}")
