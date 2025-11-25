"""
Add / overwrite an ext_roll column with custom split ratio.
Works for EG-RL: ext_full.npz  (indptr / indices / ts)
Default ratio = 0.70 / 0.15 / 0.15
"""
import argparse, pathlib, numpy as np, pandas as pd, pickle, gzip, sys

def load_df(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix == ".npz":
        arr = np.load(path, allow_pickle=True)
        if {"indptr", "indices", "ts"}.issubset(arr.files):
            indptr, indices, ts = arr["indptr"], arr["indices"], arr["ts"]
            src = np.repeat(np.arange(len(indptr) - 1), indptr[1:] - indptr[:-1])
            dst, tim = indices, ts
            return pd.DataFrame(dict(src=src, dst=dst, time=tim, val=0))
        sys.exit(f"[ERR] Unknown keys in {path.name}: {arr.files}")
    if path.suffix == ".gz":
        return pickle.load(gzip.open(path, "rb"))
    return pickle.load(open(path, "rb"))

def save_df(df: pd.DataFrame, src: pathlib.Path):
    out = src.with_suffix(".pkl")
    pickle.dump(df, open(out, "wb"))
    print(f"[SAVE] {out.relative_to(src.parent.parent)}")

def process_dataset(root: pathlib.Path, ratios):
    cand = [root / "ext_full.npz.pkl",
            root / "ext_full.npz",
            root / "df.pkl",
            root / "df.pkl.gz"]
    p = next((c for c in cand if c.exists()), None)
    if p is None:
        print(f"[SKIP] {root.name:10}  no ext_full found")
        return
    df = load_df(p).sort_values("time").reset_index(drop=True)
    n = len(df); tr = int(n*ratios[0]); va = int(n*(ratios[0]+ratios[1]))
    df["ext_roll"] = 0
    df.loc[tr:va-1,"ext_roll"] = 1
    df.loc[va:,"ext_roll"] = 2
    save_df(df, p)
    print(f"[OK]  {root.name:10} ->",
          df.ext_roll.value_counts(normalize=True).round(3).to_dict())

def main():
    ap = argparse.ArgumentParser("write ext_roll by ratio")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", help="dataset name under DATA/")
    g.add_argument("--all", action="store_true", help="process every dataset")
    ap.add_argument("--ratio", nargs=3, type=float, default=(0.7,0.15,0.15),
                    metavar=("TRAIN","VALID","TEST"))
    args = ap.parse_args()
    if not np.isclose(sum(args.ratio),1.0):
        sys.exit("ratios must sum to 1.0")
    base = pathlib.Path("DATA")
    if args.data:
        process_dataset(base/args.data.upper(), args.ratio)
    else:
        for sub in base.iterdir():
            if sub.is_dir():
                process_dataset(sub, args.ratio)

if __name__ == "__main__":
    main()
