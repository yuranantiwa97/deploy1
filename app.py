from flask import Flask, render_template, request
import pandas as pd
import os
from collections import Counter

# ====== setup dasar ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

def rupiah(v):
    try:
        return "Rp " + f"{int(float(v)):,}".replace(",", ".")
    except Exception:
        return str(v)

# daftarkan filter ke Jinja
app.jinja_env.filters["rupiah"] = rupiah

# ====== data ======
CSV_COL_PRICE = "Harga Rata-Rata Makanan di Toko (Rp)"
CSV_COL_RATING = "Rating Toko"
CSV_COL_NAME = "Nama Restoran"
CSV_COL_PREF = "Preferensi Makanan"
CSV_COL_CLUSTER = "cluster_kmeans"

csv_path = os.path.join(BASE_DIR, "data_with_cluster.csv")
df = pd.read_csv(csv_path)

# batas slider global
_prices_num = pd.to_numeric(df[CSV_COL_PRICE], errors="coerce")
_ratings_num = pd.to_numeric(df[CSV_COL_RATING], errors="coerce")
HMIN, HMAX = int(_prices_num.min()), int(_prices_num.max())
RMIN, RMAX = float(_ratings_num.min()), float(_ratings_num.max())

# list jenis (sekali hitung)
_all_jenis = set()
for s in df[CSV_COL_PREF].dropna():
    for item in str(s).split(","):
        _all_jenis.add(item.strip())
JENIS_LIST = sorted(_all_jenis)

# ====== Friendly Cluster Labels (tier saja) ======
_prices_clean = _prices_num.dropna()
q1, q2 = _prices_clean.quantile([0.33, 0.66])

def _tier_name(v):
    if v <= q1:  return "Hemat"
    if v <= q2:  return "Menengah"
    return "Premium"

_summary = (
    df.groupby(CSV_COL_CLUSTER, as_index=False)
      .agg(harga_mean=(CSV_COL_PRICE, "mean"))
)

id_to_label = {}
for _, row in _summary.iterrows():
    cid = int(row[CSV_COL_CLUSTER])
    id_to_label[cid] = _tier_name(row["harga_mean"])
    tier_order = {"Hemat": 0, "Sedang": 1, "Premium": 2}
    tier_options = sorted(set(id_to_label.values()), key=lambda t: tier_order.get(t, 99))


cluster_options = [{"id": cid, "label": lab}
                   for cid, lab in sorted(id_to_label.items(), key=lambda x: x[0])]
# ==================================================

@app.route("/", methods=["GET", "POST"])
def index():
    # ambil input
    nama = (request.values.get("nama") or "").strip().lower()
    jenis = request.values.get("jenis") or ""
    harga_min = int(float(request.values.get("harga_min") or HMIN))
    harga_max = int(float(request.values.get("harga_max") or HMAX))
    rating_min = float(request.values.get("rating_min") or RMIN)
    rating_max = float(request.values.get("rating_max") or RMAX)

    # normalisasi slider
    if harga_min > harga_max: harga_min, harga_max = harga_max, harga_min
    if rating_min > rating_max: rating_min, rating_max = rating_max, rating_min

    # cluster pilihan (pakai cluster_id; fallback ke 'cluster' lama)
    cluster_label = (request.values.get("cluster_label") or "").strip()  # pakai label, bukan ID
    cluster_id_val = (request.values.get("cluster") or "").strip()       # fallback lama (opsional)

    # filter
    hasil = df.copy()
    if nama:
        hasil = hasil[hasil[CSV_COL_NAME].fillna("").str.lower().str.contains(nama)]
    if jenis:
        hasil = hasil[hasil[CSV_COL_PREF].fillna("").str.contains(jenis, case=False)]

    harga_num = pd.to_numeric(hasil[CSV_COL_PRICE], errors="coerce")
    rating_num = pd.to_numeric(hasil[CSV_COL_RATING], errors="coerce")
    hasil = hasil[
        (harga_num >= harga_min) & (harga_num <= harga_max) &
        (rating_num >= rating_min) & (rating_num <= rating_max)
    ]

    if cluster_label:
        # filter semua cluster yang labelnya sama
        target_ids = [cid for cid, lab in id_to_label.items() if lab == cluster_label]
        hasil = hasil[hasil[CSV_COL_CLUSTER].isin(target_ids)]
    elif cluster_id_val.isdigit():
        # kompatibel lama: kalau user masih kirim ?cluster=1
        hasil = hasil[hasil[CSV_COL_CLUSTER] == int(cluster_id_val)]


    return render_template(
    "index.html",
    hasil=hasil.to_dict("records"),
    jenis_list=JENIS_LIST,

    # kirim opsi label unik & pilihan yang terpilih
    tier_options=tier_options,
    selected_cluster_label=cluster_label,

    # untuk menampilkan label di tabel
    label_map=id_to_label,
        # slider + nilai terpilih
        harga_min=HMIN, harga_max=HMAX,
        harga_min_req=harga_min, harga_max_req=harga_max,
        rating_min=RMIN, rating_max=RMAX,
        rating_min_req=rating_min, rating_max_req=rating_max,

        request=request,
    )

@app.get("/health")
def health():
    return {"ok": True, "rows": int(len(df))}

if __name__ == "__main__":
    app.run(debug=True)
