from flask import Flask, render_template, request
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

csv_path = os.path.join(BASE_DIR, "data_with_cluster.csv")
df = pd.read_csv(csv_path)

@app.route('/', methods=['GET'])
def index():
    # Filter param dari form
    nama = request.args.get('nama', '').strip().lower()
    jenis = request.args.get('jenis', '')
    harga_min = int(float(request.args.get('harga_min', df['Harga Rata-Rata Makanan di Toko (Rp)'].min())))
    harga_max = int(float(request.args.get('harga_max', df['Harga Rata-Rata Makanan di Toko (Rp)'].max())))
    rating_min = float(request.args.get('rating_min', df['Rating Toko'].min()))
    rating_max = float(request.args.get('rating_max', df['Rating Toko'].max()))
    cluster = request.args.get('cluster', '')

    hasil = df.copy()
    if nama:
        hasil = hasil[hasil['Nama Restoran'].str.lower().str.contains(nama)]
    if jenis:
        hasil = hasil[hasil['Preferensi Makanan'].str.contains(jenis, case=False, na=False)]
    hasil = hasil[
        (hasil['Harga Rata-Rata Makanan di Toko (Rp)'] >= harga_min) &
        (hasil['Harga Rata-Rata Makanan di Toko (Rp)'] <= harga_max) &
        (hasil['Rating Toko'] >= rating_min) &
        (hasil['Rating Toko'] <= rating_max)
    ]
    if cluster != '':
        hasil = hasil[hasil['cluster_kmeans'] == int(cluster)]

    # Jenis makanan 
    all_jenis = set()
    for s in df['Preferensi Makanan'].dropna():
        for item in str(s).split(','):
            all_jenis.add(item.strip())
    list_jenis = sorted(list(all_jenis))

    clusters = sorted(df['cluster_kmeans'].unique())

    return render_template(
        'index.html',
        hasil=hasil.to_dict('records'),
        jenis_list=list_jenis,
        clusters=clusters,
        harga_min=int(df['Harga Rata-Rata Makanan di Toko (Rp)'].min()),
        harga_max=int(df['Harga Rata-Rata Makanan di Toko (Rp)'].max()),
        harga_min_req=harga_min,
        harga_max_req=harga_max,
        rating_min=float(df['Rating Toko'].min()),
        rating_max=float(df['Rating Toko'].max()),
        rating_min_req=rating_min,
        rating_max_req=rating_max,
        request=request
    )

if __name__ == '__main__':
    app.run(debug=True)
