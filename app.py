from flask import Flask, render_template, request, redirect, url_for
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import os, io, base64

# === optional plotting (boleh tidak diinstall) ===
try:
    import matplotlib
    matplotlib.use("Agg")  # non-GUI backend
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# ===== setup dasar =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)

def rupiah(v):
    try:
        return "Rp " + f"{int(float(v)):,}".replace(",", ".")
    except Exception:
        return str(v)
app.jinja_env.filters["rupiah"] = rupiah

# ===== nama kolom di CSV =====
CSV_COL_PRICE   = "Harga Rata-Rata Makanan di Toko (Rp)"
CSV_COL_RATING  = "Rating Toko"
CSV_COL_NAME    = "Nama Restoran"
CSV_COL_PREF    = "Preferensi Makanan"
CSV_COL_CLUSTER = "cluster_kmeans"  # dipakai untuk label cluster dari algoritma apa pun

# ===== load data =====
csv_path = os.path.join(BASE_DIR, "data_with_cluster.csv")
df = pd.read_csv(csv_path)

# ---- kolom numerik aman ----
df["_price_num"]  = pd.to_numeric(df[CSV_COL_PRICE],  errors="coerce")
df["_rating_num"] = pd.to_numeric(df[CSV_COL_RATING], errors="coerce")

# batas slider (pakai nilai non-NaN)
_prices_num  = df["_price_num"].dropna()
_ratings_num = df["_rating_num"].dropna()
HMIN, HMAX = (int(_prices_num.min()), int(_prices_num.max())) if len(_prices_num) else (0, 0)
RMIN, RMAX = (float(_ratings_num.min()), float(_ratings_num.max())) if len(_ratings_num) else (0.0, 5.0)

# list jenis (sekali hitung)
_all_jenis = set()
for s in df[CSV_COL_PREF].dropna():
    for item in str(s).split(","):
        _all_jenis.add(item.strip())
JENIS_LIST = sorted(_all_jenis)

# ===== Kategori harga per RESTORAN (kuantil; bukan dari cluster) =====
if len(_prices_num) >= 3:
    q1, q2 = _prices_num.quantile([0.33, 0.66])
else:
    q1 = _prices_num.min() if len(_prices_num) else 0
    q2 = _prices_num.max() if len(_prices_num) else 0

def _tier_name(v):
    if pd.isna(v):  return None
    if v <= q1:  return "Hemat"
    if v <= q2:  return "Menengah"
    return "Premium"

df["tier_label"] = df["_price_num"].apply(_tier_name)
tier_options = ["Hemat", "Menengah", "Premium"]

# ===== Status training & metrik =====
TRAINED = False
METRICS = {
    "silhouette": None, "sil_quality": None, "sil_badge": None,
    "dbi": None, "dbi_quality": None, "dbi_badge": None,
    "n": 0, "algo": None, "k": None,
    "k_mode": None, "best_k": None, "k_grid": None,
    "profiles": None, "cluster_name_map": None,
    "scatter_b64": None,
    # untuk mode Auto: simpan hasil per-k di sini
    "by_k": {}  # {k: {"profiles":[...], "cluster_name_map":{...}, "scatter_b64":"...", "silhouette":..., "dbi":...}}
}
df[CSV_COL_CLUSTER] = pd.Series([pd.NA] * len(df), dtype="Int64")  # kosong di awal

# ===== util kualitas evaluasi =====
def label_silhouette(s):
    if s is None: return ("n/a", "secondary")
    if s < 0.20:  return ("Kurang baik", "danger")
    if s < 0.50:  return ("Cukup", "warning")
    if s < 0.70:  return ("Baik", "success")
    return ("Sangat baik", "success")

def label_dbi(d):
    if d is None: return ("n/a", "secondary")
    if d < 0.50:  return ("Sangat baik", "success")
    if d < 1.00:  return ("Baik", "success")
    if d < 1.50:  return ("Cukup", "warning")
    return ("Kurang baik", "danger")

# ===== util filter =====
def get_filtered_df(params):
    nama       = (params.get("nama") or "").strip().lower()
    jenis      = params.get("jenis") or ""
    harga_min  = int(float(params.get("harga_min") or HMIN))
    harga_max  = int(float(params.get("harga_max") or HMAX))
    rating_min = float(params.get("rating_min") or RMIN)
    rating_max = float(params.get("rating_max") or RMAX)
    cluster_label = (params.get("cluster_label") or "").strip()
    cluster_id_val = (params.get("cluster") or "").strip()

    if harga_min > harga_max:   harga_min, harga_max = harga_max, harga_min
    if rating_min > rating_max: rating_min, rating_max = rating_max, rating_min

    subset = df.copy()
    if nama:
        subset = subset[subset[CSV_COL_NAME].fillna("").str.lower().str.contains(nama)]
    if jenis:
        subset = subset[subset[CSV_COL_PREF].fillna("").str.contains(jenis, case=False)]
    harga_num  = pd.to_numeric(subset[CSV_COL_PRICE],  errors="coerce")
    rating_num = pd.to_numeric(subset[CSV_COL_RATING], errors="coerce")
    subset = subset[
        (harga_num >= harga_min) & (harga_num <= harga_max) &
        (rating_num >= rating_min) & (rating_num <= rating_max)
    ]
    if cluster_label:
        subset = subset[subset["tier_label"] == cluster_label]
    elif cluster_id_val.isdigit() and CSV_COL_CLUSTER in subset.columns:
        subset = subset[subset[CSV_COL_CLUSTER] == int(cluster_id_val)]

    return subset, {"harga_min": harga_min, "harga_max": harga_max,
                    "rating_min": rating_min, "rating_max": rating_max,
                    "cluster_label": cluster_label}

# ===== util visualisasi =====
def scatter_to_b64(X_orig, labels, centers_orig=None, title=None):
    if not HAVE_MPL or X_orig is None or labels is None:
        return None
    try:
        fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=110)
        ax.scatter(X_orig[:, 0], X_orig[:, 1], c=labels, s=22, alpha=0.85)
        if centers_orig is not None:
            ax.scatter(centers_orig[:, 0], centers_orig[:, 1], marker="x", s=80, linewidths=2)
        ax.set_xlabel("Harga (Rp)")
        ax.set_ylabel("Rating")
        if title: ax.set_title(title)
        ax.ticklabel_format(style="plain", axis="x")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None

# ===== helper nama manusiawi cluster berdasar mean harga =====
def human_names_for_clusters(count):
    if count <= 0: return []
    if count == 1: return ["Segmen 1"]
    if count == 2: return ["Hemat", "Premium"]
    base = ["Hemat", "Menengah", "Premium"]
    if count <= 3: return base[:count]
    return base + [f"Segmen {i}" for i in range(4, count + 1)]

# ===== training (KMeans / MeanShift) di subset =====
def train_on_subset(subset_index, algo="kmeans", k=3, k_auto=False):
    global TRAINED, METRICS, df

    # reset kolom cluster
    df[CSV_COL_CLUSTER] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # mask subset valid
    mask_subset = df.index.isin(subset_index)
    mask_valid  = df["_price_num"].notna() & df["_rating_num"].notna()
    mask_train  = mask_subset & mask_valid

    X_orig = df.loc[mask_train, ["_price_num", "_rating_num"]].to_numpy()
    if len(X_orig) < 3:
        TRAINED = False
        METRICS.update({
            "silhouette": None, "sil_quality": None, "sil_badge": None,
            "dbi": None, "dbi_quality": None, "dbi_badge": None,
            "n": int(len(X_orig)), "algo": algo, "k": (k if algo == "kmeans" else None),
            "k_mode": "manual", "best_k": None, "k_grid": None,
            "profiles": None, "cluster_name_map": None, "scatter_b64": None,
            "by_k": {}
        })
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(X_orig)

    labels, centers_std, k_used, k_grid = None, None, (k if algo == "kmeans" else None), None
    by_k = {}

    if algo == "kmeans":
        if k_auto:
            # loop k=2..6 -> simpan semua hasil ke by_k
            k_min, k_max = 2, 6
            grid = []
            best_k, best_s = None, -1.0
            for k_i in range(k_min, k_max + 1):
                km = KMeans(n_clusters=k_i, init="k-means++", n_init=10, random_state=0)
                lbl = km.fit_predict(X)
                uniq = len(set(lbl))
                if uniq >= 2 and len(X) > uniq:
                    s = float(silhouette_score(X, lbl))
                    d = float(davies_bouldin_score(X, lbl))
                else:
                    s, d = None, None
                centers = scaler.inverse_transform(km.cluster_centers_)

                # siapkan profil + nama
                profiles = []
                for cid in range(len(centers)):
                    p, r = centers[cid]
                    size = int((lbl == cid).sum())
                    profiles.append({"id": cid, "mean_price": float(p), "mean_rating": float(r), "n": size})
                ordered = sorted(profiles, key=lambda x: x["mean_price"])
                name_map = {p["id"]: human_names_for_clusters(len(ordered))[i] for i, p in enumerate(ordered)}

                sc_b64 = scatter_to_b64(
                    X_orig, lbl, centers_orig=centers,
                    title=f"K-Means (k={k_i}) · n={len(X_orig)}"
                )

                by_k[k_i] = {
                    "profiles": profiles,
                    "cluster_name_map": name_map,
                    "scatter_b64": sc_b64,
                    "silhouette": s,
                    "dbi": d,
                }
                grid.append({"k": k_i, "silhouette": s, "dbi": d})

                # pilih best_k berdasarkan Silhouette terbesar (tie-break pakai DBI terkecil)
                if s is not None:
                    if (best_k is None) or (s > best_s) or (
                        abs(s - best_s) < 1e-6 and d is not None and
                        by_k.get(best_k, {}).get("dbi") is not None and d < by_k[best_k]["dbi"]
                    ):
                        best_s = s; best_k = k_i

            # pakai best_k untuk menulis label ke dataframe
            k_used = best_k or 3
            km = KMeans(n_clusters=k_used, init="k-means++", n_init=10, random_state=0)
            labels = km.fit_predict(X)
            centers_std = km.cluster_centers_
            k_grid = grid

        else:
            k_used = max(2, int(k or 3))
            km = KMeans(n_clusters=k_used, init="k-means++", n_init=10, random_state=0)
            labels = km.fit_predict(X)
            centers_std = km.cluster_centers_

    else:
        bw = estimate_bandwidth(X, quantile=0.3, n_samples=min(500, len(X)))
        ms = MeanShift(bandwidth=bw, bin_seeding=True)
        labels = ms.fit_predict(X)
        centers_std = ms.cluster_centers_

    # tulis label hanya pada subset yang dilatih
    df.loc[mask_train, CSV_COL_CLUSTER] = labels
    df[CSV_COL_CLUSTER] = df[CSV_COL_CLUSTER].astype("Int64")

    # metrik keseluruhan (untuk k_used)
    sil, dbi = None, None
    uniq = len(set(labels))
    if uniq >= 2 and len(X) > uniq:
        sil = float(silhouette_score(X, labels))
        dbi = float(davies_bouldin_score(X, labels))
    sil_q, dbi_q = label_silhouette(sil), label_dbi(dbi)

    # profil/label untuk k_used (non-auto atau best_k)
    centers = StandardScaler().fit(X_orig).inverse_transform(centers_std) \
              if centers_std is not None else None
    profiles = []
    if centers is not None:
        for cid in range(len(centers)):
            p, r = centers[cid]
            size = int((labels == cid).sum())
            profiles.append({"id": cid, "mean_price": float(p), "mean_rating": float(r), "n": size})
        ordered = sorted(profiles, key=lambda x: x["mean_price"])
        name_map = {p["id"]: human_names_for_clusters(len(ordered))[i] for i, p in enumerate(ordered)}
    else:
        name_map = {}

    # visual (opsional)
    centers_used = None
    try:
        centers_used = StandardScaler().fit(X_orig).inverse_transform(centers_std)
    except Exception:
        pass
    scatter_b64 = scatter_to_b64(
        X_orig, labels, centers_orig=centers_used,
        title=f"{'K-Means (k='+str(k_used)+')' if algo=='kmeans' else 'MeanShift'} · n={len(X_orig)}"
    )

    METRICS.update({
        "silhouette": sil, "sil_quality": sil_q[0], "sil_badge": sil_q[1],
        "dbi": dbi, "dbi_quality": dbi_q[0], "dbi_badge": dbi_q[1],
        "n": int(len(X_orig)), "algo": algo, "k": (k_used if algo == 'kmeans' else None),
        "k_mode": ("auto" if (algo == "kmeans" and k_auto) else "manual"),
        "best_k": (k_used if algo == "kmeans" else None), "k_grid": k_grid,
        "profiles": profiles, "cluster_name_map": name_map,
        "scatter_b64": scatter_b64,
        "by_k": by_k if (algo == "kmeans" and k_auto) else {}
    })
    TRAINED = True

# ===== Routes =====
@app.get("/")
def index():
    subset, used = get_filtered_df(request.values)

    # pilihan algoritma & k (untuk state form)
    algo = (request.values.get("algo") or "kmeans").lower()
    algo = "meanshift" if algo == "meanshift" else "kmeans"
    k = int(request.values.get("k") or 3)
    k_auto = (request.values.get("k_auto") == "1")

    # === PAGINATION ===
    limit = int(request.values.get("limit") or 10)
    page  = max(1, int(request.values.get("page") or 1))
    total_count = int(len(subset))
    start = (page - 1) * limit
    end   = min(start + limit, total_count)
    hasil_view = subset.iloc[start:end]

    # siapkan URL prev/next dgn mempertahankan query
    args_now = request.values.to_dict(flat=True)
    args_now["limit"] = limit
    prev_url = url_for("index", **{**args_now, "page": page-1}) if start > 0 else None
    next_url = url_for("index", **{**args_now, "page": page+1}) if end < total_count else None

    # === VIEW-K untuk Auto ===
    list_k = sorted((METRICS.get("by_k") or {}).keys())
    try:
        view_k = int(request.values.get("view_k")) if request.values.get("view_k") else None
    except Exception:
        view_k = None

    if METRICS.get("k_mode") == "auto" and list_k:
        if view_k is None:
            view_k = METRICS.get("best_k")

        row = METRICS["by_k"].get(view_k, {}) or {}
        sil = row.get("silhouette")
        dbi = row.get("dbi")
        sq = label_silhouette(sil)
        dq = label_dbi(dbi)

        metrics_view = {
            "profiles": row.get("profiles") or METRICS.get("profiles"),
            "cluster_name_map": row.get("cluster_name_map") or METRICS.get("cluster_name_map"),
            "scatter_b64": row.get("scatter_b64") or METRICS.get("scatter_b64"),
            "silhouette": sil, "dbi": dbi,
            "sil_quality": sq[0], "sil_badge": sq[1],
            "dbi_quality": dq[0], "dbi_badge": dq[1],
            "k_view": view_k,
        }
    else:
        metrics_view = {
            "profiles": METRICS.get("profiles"),
            "cluster_name_map": METRICS.get("cluster_name_map"),
            "scatter_b64": METRICS.get("scatter_b64"),
            "silhouette": METRICS.get("silhouette"),
            "dbi": METRICS.get("dbi"),
            "sil_quality": METRICS.get("sil_quality"),
            "sil_badge": METRICS.get("sil_badge"),
            "dbi_quality": METRICS.get("dbi_quality"),
            "dbi_badge": METRICS.get("dbi_badge"),
            "k_view": METRICS.get("k"),
        }

    return render_template(
        "index.html",
        hasil=hasil_view.to_dict("records"),
        total_count=total_count, limit=limit, page=page,
        start_idx=(start + 1 if total_count else 0), end_idx=end,
        prev_url=prev_url, next_url=next_url,

        trained=TRAINED, metrics=METRICS,      # ringkasan global (tetap dikirim)
        metrics_view=metrics_view,             # metrik yang tampil (per-k saat Auto)
        list_k=list_k, view_k=view_k,

        algo=algo, k=k, k_auto=k_auto,

        jenis_list=JENIS_LIST,
        tier_options=tier_options, selected_cluster_label=used["cluster_label"],

        harga_min=HMIN, harga_max=HMAX,
        harga_min_req=used["harga_min"], harga_max_req=used["harga_max"],
        rating_min=RMIN, rating_max=RMAX,
        rating_min_req=used["rating_min"], rating_max_req=used["rating_max"],

        request=request,
    )

@app.post("/train")
def train():
    subset, _ = get_filtered_df(request.values)
    algo = (request.values.get("algo") or "kmeans").lower()
    algo = "meanshift" if algo == "meanshift" else "kmeans"
    k = int(request.values.get("k") or 3)
    k_auto = (request.values.get("k_auto") == "1")
    train_on_subset(subset.index, algo=algo, k=k, k_auto=k_auto)
    return redirect(url_for("index", **request.values))

@app.post("/reset")
def reset():
    global TRAINED, METRICS, df
    df[CSV_COL_CLUSTER] = pd.Series([pd.NA] * len(df), dtype="Int64")
    TRAINED = False
    METRICS = {
        "silhouette": None, "sil_quality": None, "sil_badge": None,
        "dbi": None, "dbi_quality": None, "dbi_badge": None,
        "n": 0, "algo": None, "k": None,
        "k_mode": None, "best_k": None, "k_grid": None,
        "profiles": None, "cluster_name_map": None,
        "scatter_b64": None,
        "by_k": {}
    }
    return redirect(url_for("index", **request.values))

@app.get("/health")
def health():
    return {"ok": True, "rows": int(len(df)), "trained": TRAINED, "metrics": METRICS}

# no-cache agar tidak menampilkan HTML lama
@app.after_request
def add_no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
