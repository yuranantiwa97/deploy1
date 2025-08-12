from flask import Flask, render_template, request, redirect, url_for
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import os

# ===== setup dasar =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

def rupiah(v):
    try:
        return "Rp " + f"{int(float(v)):,}".replace(",", ".")
    except Exception:
        return str(v)

# daftar filter ke Jinja (dipakai di template)
app.jinja_env.filters["rupiah"] = rupiah

# ===== nama kolom di CSV =====
CSV_COL_PRICE   = "Harga Rata-Rata Makanan di Toko (Rp)"
CSV_COL_RATING  = "Rating Toko"
CSV_COL_NAME    = "Nama Restoran"
CSV_COL_PREF    = "Preferensi Makanan"
CSV_COL_CLUSTER = "cluster_kmeans"  # akan dibuat/ditimpa

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

# ===== Kategori harga per RESTORAN (bukan per cluster) =====
if len(_prices_num) >= 3:
    q1, q2 = _prices_num.quantile([0.33, 0.66])
else:
    q1 = _prices_num.min() if len(_prices_num) else 0
    q2 = _prices_num.max() if len(_prices_num) else 0

def _tier_name(v):
    if pd.isna(v):  # aman untuk NaN
        return None
    if v <= q1:  return "Hemat"
    if v <= q2:  return "Menengah"
    return "Premium"

df["tier_label"] = df["_price_num"].apply(_tier_name)
tier_options = ["Hemat", "Menengah", "Premium"]

# ===== Status training & metrik (global sederhana) =====
TRAINED = False
METRICS = {"silhouette": None, "dbi": None, "n": 0}

# Pastikan kolom cluster ada & kosong (SEBELUM training)
df[CSV_COL_CLUSTER] = pd.Series([pd.NA] * len(df), dtype="Int64")

def train_kmeans():
    """Latih KMeans pada kolom harga & rating, isi cluster_kmeans + hitung metrik."""
    global TRAINED, METRICS, df

    mask_train = df["_price_num"].notna() & df["_rating_num"].notna()
    X = df.loc[mask_train, ["_price_num", "_rating_num"]].to_numpy()

    # Guard data minimal
    if len(X) < 3:
        TRAINED = False
        METRICS = {"silhouette": None, "dbi": None, "n": int(len(X))}
        # kosongkan cluster
        df[CSV_COL_CLUSTER] = pd.Series([pd.NA] * len(df), dtype="Int64")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=0)
    labels = kmeans.fit_predict(X_scaled)

    # tulis hasil ke kolom cluster_kmeans (nullable Int64)
    df[CSV_COL_CLUSTER] = pd.NA
    df.loc[mask_train, CSV_COL_CLUSTER] = labels
    df[CSV_COL_CLUSTER] = df[CSV_COL_CLUSTER].astype("Int64")

    # hitung metrik (butuh >=2 cluster & sampel > jumlah cluster)
    unique_clusters = pd.Series(labels).nunique()
    sil, dbi = None, None
    if unique_clusters >= 2 and len(X_scaled) > unique_clusters:
        sil = float(silhouette_score(X_scaled, labels))
        dbi = float(davies_bouldin_score(X_scaled, labels))

    METRICS = {"silhouette": sil, "dbi": dbi, "n": int(len(X))}
    TRAINED = True

@app.route("/", methods=["GET"])
def index():
    # ambil input
    nama       = (request.values.get("nama") or "").strip().lower()
    jenis      = request.values.get("jenis") or ""
    harga_min  = int(float(request.values.get("harga_min") or HMIN))
    harga_max  = int(float(request.values.get("harga_max") or HMAX))
    rating_min = float(request.values.get("rating_min") or RMIN)
    rating_max = float(request.values.get("rating_max") or RMAX)

    # normalisasi slider bila ketuker
    if harga_min > harga_max:   harga_min,  harga_max  = harga_max,  harga_min
    if rating_min > rating_max: rating_min, rating_max = rating_max, rating_min

    # baca kategori (label unik)
    cluster_label = (request.values.get("cluster_label") or "").strip()

    # fallback lama (?cluster=1) — tetap didukung tapi opsional
    cluster_id_val = (request.values.get("cluster") or "").strip()

    # mulai filter
    hasil = df.copy()

    if nama:
        hasil = hasil[hasil[CSV_COL_NAME].fillna("").str.lower().str.contains(nama)]
    if jenis:
        hasil = hasil[hasil[CSV_COL_PREF].fillna("").str.contains(jenis, case=False)]

    harga_num  = pd.to_numeric(hasil[CSV_COL_PRICE],  errors="coerce")
    rating_num = pd.to_numeric(hasil[CSV_COL_RATING], errors="coerce")
    hasil = hasil[
        (harga_num  >= harga_min)  & (harga_num  <= harga_max) &
        (rating_num >= rating_min) & (rating_num <= rating_max)
    ]

    # filter berdasar kategori harga per RESTORAN
    if cluster_label:
        hasil = hasil[hasil["tier_label"] == cluster_label]
    # filter lama berdasar ID cluster (optional) — hanya jika kolom ada & digit
    elif cluster_id_val.isdigit() and CSV_COL_CLUSTER in hasil.columns:
        hasil = hasil[hasil[CSV_COL_CLUSTER] == int(cluster_id_val)]

    # ===== Batasi 10 baris =====
    total_count = int(len(hasil))
    hasil_view = hasil.head(10)

    return render_template(
        "index.html",
        hasil=hasil_view.to_dict("records"),
        total_count=total_count,
        limit=10,
        trained=TRAINED,
        metrics=METRICS,
        jenis_list=JENIS_LIST,

        # dropdown kategori (label unik)
        tier_options=tier_options,
        selected_cluster_label=cluster_label,

        # slider + nilai terpilih
        harga_min=HMIN, harga_max=HMAX,
        harga_min_req=harga_min, harga_max_req=harga_max,
        rating_min=RMIN, rating_max=RMAX,
        rating_min_req=rating_min, rating_max_req=rating_max,

        request=request,
    )

@app.post("/train")
def train():
    train_kmeans()
    return redirect(url_for("index"))

@app.post("/reset")
def reset():
    global TRAINED, METRICS, df
    df[CSV_COL_CLUSTER] = pd.Series([pd.NA] * len(df), dtype="Int64")
    TRAINED = False
    METRICS = {"silhouette": None, "dbi": None, "n": 0}
    return redirect(url_for("index"))

@app.get("/health")
def health():
    return {"ok": True, "rows": int(len(df)), "trained": TRAINED, "metrics": METRICS}

if __name__ == "__main__":
    app.run(debug=True)
