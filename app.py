from flask import Flask, render_template, request
import pandas as pd
import os
from collections import Counter

# --- setup dasar ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

def rupiah(v):
    try:
        return "Rp " + f"{int(float(v)):,}".replace(",", ".")
    except Exception:
        return str(v)
app.jinja_env.filters["rupiah"] = rupiah

# --- data ---
csv_path = os.path.join(BASE_DIR, "data_with_cluster.csv")
df = pd.read_csv(csv_path)

# batas global (buat default slider)
HMIN, HMAX = int(df['Harga Rata-Rata Makanan di Toko (Rp)'].min()), int(df['Harga Rata-Rata Makanan di Toko (Rp)'].max())
RMIN, RMAX = float(df['Rating Toko'].min()), float(df['Rating Toko'].max())

# jenis makanan (sekali hitung)
_all_jenis = set()
for s in df['Preferensi Makanan'].dropna():
    for item in str(s).split(','):
        _all_jenis.add(item.strip())
JENIS_LIST = sorted(_all_jenis)

# --- Friendly Cluster Labels (sekali hitung) ---
_prices = df["Harga Rata-Rata Makanan di Toko (Rp)"].astype(float)
q1, q2 = _prices.quantile([0.33, 0.66])

def _tier_name(v):
    if v <= q1:  return "Hemat"
    if v <= q2:  return "Sedang"
    return "Premium"

def _top_pref(series):
    c = Counter()
    for s in series.dropna().astype(str):
        for w in s.split(","):
            w = w.strip()
            if w:
                c[w] += 1
    return c.most_common(1)[0][0] if c else "Campuran"

_summary = (
    df.groupby("cluster_kmeans", as_index=False)
      .agg(
          harga_mean=("Harga Rata-Rata Makanan di Toko (Rp)", "mean"),
          rating_mean=("Rating Toko", "mean"),
          top_pref=("Preferensi Makanan", _top_pref),
      )
)

id_to_label = {}
for _, row in _summary.iterrows():
    cid = int(row["cluster_kmeans"])
    label = f"{_tier_name(row['harga_mean'])} • {row['top_pref']} • ⭐{row['rating_mean']:.1f} • {rupiah(row['harga_mean'])}"
    id_to_label[cid] = label

cluster_options = [{"id": cid, "label": lab} for cid, lab in sorted(id_to_label.items(), key=lambda x: x[0])]
# ------------------------------------------------

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

    # cluster ramah (label) + kompatibel lama ('cluster' angka)
    cluster_label = (request.values.get("cluster_label") or "").strip()
    selected_cluster_id = None
    if cluster_label:
        if cluster_label.isdigit():
            selected_cluster_id = int(cluster_label)
        else:
            selected_cluster_id = next((cid for cid, lab in id_to_label.items() if lab == cluster_label), None)
    else:
        _cluster_old = (request.values.get("cluster") or "").strip()
        if _cluster_old.isdigit():
            selected_cluster_id = int(_cluster_old)

    # filter
    hasil = df.copy()
    if nama:
        hasil = hasil[hasil["Nama Restoran"].str.lower().str.contains(nama, na=False)]
    if jenis:
        hasil = hasil[hasil["Preferensi Makanan"].fillna("").str.contains(jenis, case=False)]
    hasil = hasil[
        (hasil['Harga Rata-Rata Makanan di Toko (Rp)'] >= harga_min) &
        (hasil['Harga Rata-Rata Makanan di Toko (Rp)'] <= harga_max) &
        (hasil['Rating Toko'] >= rating_min) &
        (hasil['Rating Toko'] <= rating_max)
    ]
    if selected_cluster_id is not None:
        hasil = hasil[hasil["cluster_kmeans"] == selected_cluster_id]

    return render_template(
        "index.html",
        hasil=hasil.to_dict("records"),
        jenis_list=JENIS_LIST,

        # dukung lama (angka) & baru (label ramah)
        clusters=sorted(df['cluster_kmeans'].unique()),
        cluster_options=cluster_options,
        selected_cluster_label=cluster_label,
        label_map=id_to_label,

        # batas slider & nilai terpilih
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
