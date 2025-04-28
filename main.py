from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import HTMLResponse

# Inisialisasi FastAPI app
app = FastAPI()

# Load model dan scaler
model = joblib.load('kmeans_model.pkl')  # Ganti nama file model
scaler = joblib.load('scaler.pkl')       # Ganti nama file scaler

# Membuat kelas untuk input data prediksi
class PredictRequest(BaseModel):
    recency: float
    frequency: float
    monetary: float

# Mapping cluster ke label
cluster_labels = {
    0: "Pelanggan VIP",
    1: "Pelanggan Inaktif",
    2: "Pelanggan Potensial"
}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>KMeans Cluster Prediction</title>
        </head>
        <body>
            <h1>Model KMeans sudah aktif!</h1>
            <p>Untuk melakukan prediksi, kirimkan data dengan fitur berikut ke endpoint <strong>/predict/</strong> menggunakan metode POST:</p>
            <ul>
                <li><strong>Recency:</strong> Waktu dalam hari sejak transaksi terakhir (contoh: 5.0)</li>
                <li><strong>Frequency:</strong> Jumlah transaksi pelanggan (contoh: 3)</li>
                <li><strong>Monetary:</strong> Total pengeluaran pelanggan (contoh: 1000.0)</li>
            </ul>
            <p>Format input JSON yang diperlukan:</p>
            <pre>
{
    "recency": 5.0,
    "frequency": 3,
    "monetary": 1000.0
}
            </pre>
            <h2>Cluster Labels:</h2>
            <ul>
                <li><strong>0:</strong> Pelanggan VIP</li>
                <li><strong>1:</strong> Pelanggan Inaktif</li>
                <li><strong>2:</strong> Pelanggan Potensial</li>
            </ul>
            <p>Setelah data dikirim, model akan mengembalikan hasil prediksi cluster beserta labelnya.</p>
        </body>
    </html>
    """

@app.post("/predict/")
async def predict(request: PredictRequest):
    # Mengambil data dari request dan mengubahnya menjadi array 2D
    input_data = np.array([[request.recency, request.frequency, request.monetary]])
    
    # Normalisasi input data menggunakan scaler
    scaled_data = scaler.transform(input_data)
    
    # Prediksi cluster menggunakan model KMeans
    cluster = model.predict(scaled_data)[0]
    
    # Ambil label dari hasil cluster
    label = cluster_labels.get(cluster, "Cluster Tidak Diketahui")
    
    return {"cluster": int(cluster), "label": label}
