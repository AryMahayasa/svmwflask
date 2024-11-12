from flask import Flask, request, jsonify
import joblib
import requests
from flask_cors import CORS
import pandas as pd

# Load the keywords from the uploaded CSV
keywords_df = pd.read_csv('sentiment_keywords_large_id.csv')
positive_keywords = keywords_df['positive_keywords'].dropna().tolist()
negative_keywords = keywords_df['negative_keywords'].dropna().tolist()

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the saved model, vectorizer, and label encoder
loaded_model = joblib.load('svm_sentiment_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
loaded_le = joblib.load('label_encoder.pkl')

# Extended keywords for aspect categorization with more terms
extended_keywords_informal = {
    'kualitas_pengiriman': [
        'nyasar', 'gak datang', 'terlambat', 'rusak', 'muter', 'datang terlambat',
        'lama sampai', 'belum sampai', 'tunggu paket', 'barang hilang', 'barang pecah',
        'barang salah', 'pengiriman lambat', 'pengiriman cepat', 'tracking tidak update',
        'sampai', 'lama', 'cepet', 'nyasar', 'hampir', 'telat', 'kerusakan', 'tercecer',
        'proses pengiriman', 'antar lama', 'status tidak update', 'terlalu lama',
        'barang tidak ditemukan', 'tidak ada kabar', 'pengiriman tepat waktu',
        'mudah ditemukan', 'delivery oke', 'pengiriman sangat cepat', 'keterlambatan pengiriman',
        'paket rusak', 'barang tertukar', 'sampai terlambat', 'tidak tepat waktu',
        'pengiriman tidak sesuai estimasi', 'proses kirim lambat', 'pengiriman kilat',
        'pengiriman lancar', 'estimasi pengiriman tepat', 'tracking error', 'estimasi tidak jelas',
        'barang hancur', 'lama banget', 'cepat sampai', 'estimasi tidak sesuai', 'tracking tidak akurat',
        'proses terlalu lama', 'update lambat', 'kehilangan paket', 'pengiriman meleset',
        'jadwal tidak konsisten', 'paket terlantar', 'kirim ulang', 'barang cacat', 'jasa pengiriman buruk',
        'tiba rusak', 'packing kurang baik', 'pengiriman terlambat parah', 'pengiriman tidak sampai',
        'barang terjebak', 'informasi kiriman tidak jelas', 'terjebak dalam transit', 'keamanan paket rendah',
        'ekspedisi tidak kompeten', 'waktu tempuh lama', 'keterlambatan parah', 'pengiriman terganggu',
        'jntexpressid', 'paket', 'kirim', 'kurir', 'kurirnya', 'sampe', 'sampai', 'barang'
    ],
    'layanan_pelanggan': [
        'dm', 'customer service', 'respon', 'balasan', 'komunikasi', 'hubungi',
        'customer care', 'tidak ada respon', 'tidak direspon', 'pelayanan lambat',
        'tidak ditanggapi', 'tidak dibalas', 'susah dihubungi', 'hubungi cs', 'telepon cs',
        'gak direspon', 'susah', 'service buruk', 'respons lama', 'tidak membantu',
        'tidak memuaskan', 'komplain', 'tidak dijawab', 'tidak dilayani', 'respon lambat',
        'kurang perhatian', 'pelayanan ramah', 'helpful', 'staff baik', 'proaktif',
        'solusi cepat', 'pelayanan tidak memuaskan', 'balasan lama', 'tidak responsif',
        'sulit menghubungi CS', 'customer service lambat', 'tidak ada tanggapan',
        'layanan tidak profesional', 'helpdesk tidak tanggap', 'staff tidak peduli',
        'respon cepat', 'layanan kurang memadai', 'customer service lambat', 'susah dapat balasan',
        'pelayanan tidak ramah', 'sangat responsif', 'tidak ditanggapi', 'sulit berkomunikasi',
        'bantuan kurang maksimal', 'komplain tidak digubris', 'balasan memuaskan',
        'keluhan ditanggapi', 'customer care responsif', 'sangat membantu', 'tidak peka',
        'tanggapan lama', 'tidak ada solusi', 'balasan cepat', 'customer care tidak berguna',
        'keluhan diabaikan', 'interaksi customer service tidak ramah', 'tanggapan lambat',
        'respon memuaskan', 'helpdesk profesional', 'kesan ramah', 'sulit mendapatkan bantuan',
        'min', 'halo', 'mohon', 'tolong', 'bantu', 'dibantu', 'saya'
    ],
    'keandalan': [
        'keandalan', 'tepat waktu', 'janji layanan', 'sistem down', 'muter-muter',
        'tidak sesuai', 'janji tidak ditepati', 'tidak akurat', 'tidak profesional',
        'bermasalah', 'tidak memuaskan', 'pengiriman tidak jelas', 'informasi tidak akurat',
        'tidak dapat diandalkan', 'sistem sering down', 'gak jelas', 'ngaco', 'bisa dipercaya',
        'nggak bener', 'tidak sesuai harapan', 'ketidakpastian', 'tidak konsisten', 'sering gagal',
        'banyak kendala', 'sistem sering error', 'tidak bisa diandalkan', 'masalah konsistensi',
        'pengiriman tidak sesuai', 'layanan kacau', 'sering terjadi error', 'kualitas layanan rendah',
        'sangat bisa dipercaya', 'tidak stabil', 'kualitas kurang baik', 'sering salah',
        'janji kosong', 'tidak memuaskan', 'kurang konsisten', 'error berulang', 'banyak kekurangan',
        'sangat profesional', 'sering gagal kirim', 'kurang bisa diandalkan', 'tidak sesuai prosedur',
        'gangguan teknis', 'performa buruk', 'kurang memenuhi standar', 'sangat kacau', 'sering delay',
        'komitmen lemah', 'standar layanan rendah', 'pelayanan tidak konsisten', 'masalah berkelanjutan',
        'tidak ada kepastian', 'sistem sering crash', 'tidak bisa dipercaya', 'sistem tidak stabil'
    ],
    'kemudahan_penggunaan': [
        'kemudahan', 'pemesanan', 'tracking', 'pelacakan', 'pengembalian', 'aplikasi sulit',
        'pelacakan susah', 'cek resi', 'cek paket', 'cek posisi', 'tidak bisa lacak',
        'proses pengembalian', 'balikin barang', 'refund sulit', 'kirim ulang', 'gagal melacak',
        'resi tidak muncul', 'ribet', 'susah banget', 'gampang', 'cepat', 'mudah digunakan',
        'akses cepat', 'pengguna ramah', 'fitur lengkap', 'kesulitan akses', 'user friendly',
        'tidak intuitif', 'interface jelas', 'proses mudah', 'navigasi cepat', 'simpel', 'praktis',
        'aplikasi susah digunakan', 'user interface tidak user-friendly', 'cek resi tidak bekerja',
        'sulit mencari informasi', 'aplikasi rumit', 'sulit akses', 'pengembalian barang sulit',
        'tracking tidak berfungsi', 'antarmuka intuitif', 'pengembalian barang gampang',
        'navigasi aplikasi buruk', 'fitur aplikasinya lengkap', 'refund gampang', 'tampilan simpel',
        'penggunaan praktis', 'mudah dimengerti', 'proses pemesanan praktis', 'aplikasi cepat',
        'sistem ribet', 'fitur tidak lengkap', 'sangat praktis', 'langkah mudah', 'sangat user-friendly',
        'gampang banget', 'susah diakses', 'proses refund mudah', 'fitur tidak berfungsi',
        'aplikasi crash', 'simpel digunakan', 'akses gampang', 'aplikasi friendly', 'tidak ribet',
        'fitur error', 'tampilan jelas', 'cek paket sulit', 'user experience buruk',
        'alur aplikasi rumit', 'sulit navigasi', 'mudah memahami menu', 'proses mudah diikuti',
        'interface nyaman digunakan', 'aksesibilitas buruk', 'fitur intuitif'
    ]
}

# Preprocessing function
def preprocess_and_normalize_slang(text):
    return str(text).lower().strip()  # Convert to lowercase and remove extra spaces

# Aspect categorization function
def analyze_aspects(text):
    aspect_counts = {aspect: 0 for aspect in extended_keywords_informal.keys()}

    # Check each aspect for keyword matches in the text
    for aspect, keywords in extended_keywords_informal.items():
        for keyword in keywords:
            if keyword in text:
                aspect_counts[aspect] += 1

    # Determine relevant aspects
    relevant_aspects = [aspect for aspect, count in aspect_counts.items() if count > 0]
    return relevant_aspects if relevant_aspects else ["uncategorized"]

# Sentiment analysis function
def analyze_user_input(input_text):
    # Preprocess input text
    input_text = preprocess_and_normalize_slang(input_text)

    # Vectorize input text using the loaded vectorizer
    input_vectorized = loaded_vectorizer.transform([input_text])

    # Predict sentiment using the loaded model
    predicted_label = loaded_model.predict(input_vectorized)[0]
    predicted_sentiment = loaded_le.inverse_transform([predicted_label])[0]

    # Manual sentiment analysis
    sentiment_score = 0
    for keyword in positive_keywords:
        if keyword in input_text:
            sentiment_score += 1
    for keyword in negative_keywords:
        if keyword in input_text:
            sentiment_score -= 2

    manual_sentiment = (
        'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'
    )

    # Aspect analysis
    aspects = analyze_aspects(input_text)

    # Return combined results
    return {
        'predicted_sentiment': predicted_sentiment,
        'manual_sentiment': manual_sentiment,
        'aspects': aspects
    }

# Function to send results to Fuseki
def send_to_fuseki(tweet_id, manual_sentiment, predicted_sentiment, input_text, aspects):
    # Escape double quotes in the input_text to avoid parsing errors
    input_text = input_text.replace('"', '\\"')

    # Join aspects list into a single string
    aspects_str = ", ".join(aspects).replace('"', '\\"')

    fuseki_endpoint = "http://localhost:3030/sentimen/update"  # Replace with your dataset name
    sparql_insert_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX d: <http://www.semanticweb.org/arima/ontologies/2024/10/untitled-ontology-6#>

    INSERT DATA {{
      d:{tweet_id} rdf:type d:Tweet;
                   d:SentimenManual "{manual_sentiment}";
                   d:Sentimen "{predicted_sentiment}";
                   d:IsiTweet "{input_text}";
                   d:Aspek "{aspects_str}".
    }}
    """

    response = requests.post(
        fuseki_endpoint,
        data={'update': sparql_insert_query},
        headers={'Content-Type': 'application/x-www-form-urlencoded'}
    )

    if response.status_code == 200:
        print("Data successfully inserted into Fuseki.")
    else:
        print(f"Failed to insert data. Status code: {response.status_code}")
        print(response.text)

# Define a route for sentiment analysis
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({'error': 'No text field provided'}), 400

    input_text = data['text']
    result = analyze_user_input(input_text)

    # Generate a unique ID for the tweet (use a simple hash)
    tweet_id = f"tweet{abs(hash(input_text))}"

    # Send results to Fuseki
    send_to_fuseki(tweet_id, result['manual_sentiment'], result['predicted_sentiment'], input_text, result['aspects'])

    return jsonify(result)

# Main entry point to run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
