import gradio as gr
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from nltk.tokenize import word_tokenize
import nltk
import chardet
import grpc
import zemberek_grpc.morphology_pb2 as z_morphology
import zemberek_grpc.morphology_pb2_grpc as z_morphology_g
from functools import lru_cache

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords

turkish_stopwords = set(stopwords.words('turkish'))

extended_turkish_stopwords = {
    'ama', 'amma', 'anca', 'ancak', 'belki', 'çünkü', 'dahi', 'eğer', 'emme', 'fakat',
    'gah', 'gerek', 'hakeza', 'halbuki', 'hatta', 'hele', 'hem', 'hoş', 'ile', 'imdi',
    'ister', 'kah', 'keşke', 'keza', 'kezalik', 'kim', 'lakin', 'madem', 'mademki',
    'mamafih', 'meğer', 'meğerki', 'meğerse', 'netekim', 'neyse', 'nitekim', 'oysa',
    'oysaki', 'şayet', 'velev', 'velhasıl', 'velhasılıkelam', 'veya', 'veyahut', 'yahut',
    'yalnız', 'yani', 'yok', 'yoksa', 'zira', 'acaba', 'acep', 'açıkça', 'açıkçası',
    'adamakıllı', 'adeta', 'bazen', 'bazı', 'bilcümle', 'binaen', 'binaenaleyh', 'bir',
    'biraz', 'birazdan', 'birden', 'birdenbire', 'birice', 'birlikte', 'bitevi', 'biteviye',
    'bittabi', 'bizatihi', 'bizce', 'bizcileyin', 'bizden', 'bizzat', 'boşuna', 'böyle',
    'böylece', 'böylecene', 'böylelikle', 'böylemesine', 'böylesine', 'buracıkta', 'burada',
    'buradan', 'büsbütün', 'çabuk', 'çabukça', 'çeşitli', 'çoğu', 'çoğun', 'çoğunca',
    'çoğunlukla', 'çok', 'çokça', 'çokluk', 'çoklukla', 'cuk', 'daha', 'dahil', 'dahilen',
    'daima', 'demin', 'demincek', 'deminden', 'derakap', 'derhal', 'derken', 'diye', 'elbet',
    'elbette', 'enikonu', 'epey', 'epeyce', 'epeyi', 'esasen', 'esnasında', 'etraflı',
    'etraflıca', 'evleviyetle', 'evvel', 'evvela', 'evvelce', 'evvelden', 'evvelemirde',
    'evveli', 'gayet', 'gayetle', 'gayri', 'gayrı', 'geçende', 'geçenlerde', 'gene', 'gerçi',
    'gibi', 'gibilerden', 'gibisinden', 'gine', 'halen', 'halihazırda', 'haliyle', 'handiyse',
    'hani', 'hasılı', 'hulasaten', 'iken', 'illa', 'illaki', 'itibarıyla', 'iyice', 'iyicene',
    'kala', 'kez', 'kısaca', 'külliyen', 'lütfen', 'nasıl', 'nasılsa', 'nazaran', 'neden',
    'nedeniyle', 'nedense', 'nerde', 'nerden', 'nerdeyse', 'nerede', 'nereden', 'neredeyse',
    'nereye', 'neye', 'neyi', 'nice', 'niçin', 'nihayet', 'nihayetinde', 'niye', 'oldu',
    'oldukça', 'olur', 'onca', 'önce', 'önceden', 'önceleri', 'öncelikle', 'onculayın',
    'ondan', 'oracık', 'oracıkta', 'orada', 'oradan', 'oranca', 'oranla', 'oraya', 'öyle',
    'öylece', 'öylelikle', 'öylemesine', 'pek', 'pekala', 'pekçe', 'peki', 'peyderpey',
    'sadece', 'sahi', 'sahiden', 'sanki', 'sonra', 'sonradan', 'sonraları', 'sonunda',
    'şöyle', 'şuncacık', 'şuracıkta', 'tabii', 'tam', 'tamamen', 'tamamıyla',
    'tek', 'vasıtasıyla', 'yakinen', 'yakında', 'yakından', 'yakınlarda', 'yalnız',
    'yalnızca', 'yeniden', 'yenilerde', 'yine', 'yok', 'yoluyla', 'yüzünden', 'zaten',
    'zati', 'ait', 'bari', 'beri', 'bile', 'değin', 'dek', 'denli', 'doğru', 'dolayı',
    'dolayısıyla', 'gelgelelim', 'gibi', 'gırla', 'göre', 'hasebiyle', 'için', 'ila',
    'ile', 'ilen', 'indinde', 'inen', 'kadar', 'kaffesi', 'karşın', 'kelli', 'leh',
    'maada', 'mebni', 'naşi', 'rağmen', 'üzere', 'zarfında', 'öbür', 'bana', 'başkası',
    'ben', 'beriki', 'birbiri', 'birçoğu', 'biri', 'birileri', 'birisi', 'birkaçı', 'biz',
    'bizimki', 'buna', 'bunda', 'bundan', 'bunlar', 'bunu', 'bunun', 'burası', 'çoğu',
    'çokları', 'çoklarınca', 'cümlesi', 'değil', 'diğeri', 'filanca', 'hangisi', 'hepsi',
    'hiçbiri', 'iş', 'kaçı', 'kaynak', 'kendi', 'kim', 'kimi', 'kimisi', 'kimse',
    'kimsecik', 'kimsecikler', 'nere', 'neresi', 'öbürkü', 'öbürü', 'ona', 'onda',
    'ondan', 'onlar', 'onu', 'onun', 'öteki', 'ötekisi', 'öz', 'sana', 'sen', 'siz',
    'şuna', 'şunda', 'şundan', 'şunlar', 'şunu', 'şunun', 'şura', 'şuracık', 'şurası'
}

all_stopwords = turkish_stopwords.union(extended_turkish_stopwords)

# Zemberek GRPC bağlantısı
channel = grpc.insecure_channel('localhost:6789')
morphology_stub = z_morphology_g.MorphologyServiceStub(channel)

# Önbellekli kök bulma fonksiyonu
@lru_cache(maxsize=10000)
def get_stem(word):
    try:
        response = morphology_stub.AnalyzeWord(z_morphology.WordAnalysisRequest(input=word))
        if response.analyses:
            return response.analyses[0].lemmas[0]  # İlk lemma
        return word
    except grpc.RpcError as e:
        print(f"Zemberek hatası: {e}")
        return word

model_path = Path(r"C:\Users\Melek\yapayZeka\FilmTemaAnaliziProje\Models\tam_veri_seti_model4.pkl")
try:
    model = joblib.load(model_path)
    model_loaded = True
except:
    print(f"Model {model_path} konumundan yüklenemedi. Tahmin yapılmadan demo olarak çalışacaktır.")
    model_loaded = False


themes = [
    "romantik", "savaş", "bilim kurgu", "aksiyon", "dram", "fantastik", "gerilim", "suç",
    "tarih", "müzik", "komedi", "korku", "animasyon", "spor", "distoptik", "polisiye"
]


def clean_srt_file(file_content):
    lines = file_content.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\d+$', line.strip()) or '-->' in line:
            continue
        if line.strip():
            cleaned_lines.append(line.strip())
    return cleaned_lines


def split_into_sentences(lines):
    sentences = []
    for line in lines:
        parts = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\…)\s', line)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def clean_sentence(sentence):
    sentence = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ0-9]', ' ', sentence)
    sentence = sentence.lower()
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence


def remove_stopwords_and_stem(sentence):
    words = word_tokenize(sentence)
    processed_words = []
    for word in words:
        if word not in all_stopwords and (len(word) > 2 or word.isdigit() or re.match(r'^\w+-\d+\.?\d*$', word)):
            stemmed_word = get_stem(word)
            processed_words.append(stemmed_word)
    return ' '.join(processed_words)


def process_subtitle_file(srt_content):
    cleaned_lines = clean_srt_file(srt_content)
    sentences = split_into_sentences(cleaned_lines)
    processed_sentences = []
    for sentence in sentences:
        cleaned_sentence = clean_sentence(sentence)
        final_sentence = remove_stopwords_and_stem(cleaned_sentence)
        if final_sentence:  # Sadece boş olmayan cümleleri ekle
            processed_sentences.append(final_sentence)
    return processed_sentences


def create_chunks(sentences, chunk_size=20):
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i + chunk_size]
        chunks.append(' '.join(chunk))
    return chunks


def predict_theme_distribution(chunks):
    if not model_loaded:
        return {theme: np.random.uniform(0, 1) for theme in themes}

    predictions = model.predict(chunks)

    theme_counts = {}
    for theme in themes:
        theme_counts[theme] = 0

    for pred in predictions:
        theme_counts[pred] = theme_counts.get(pred, 0) + 1

    total = len(predictions)
    theme_percentages = {theme: (count / total) * 100 for theme, count in theme_counts.items()}

    return theme_percentages


def plot_theme_distribution(theme_percentages):
    sorted_themes = sorted(theme_percentages.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_themes]
    values = [item[1] for item in sorted_themes]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, values, color='skyblue')
    plt.xlabel('Film Temaları')
    plt.ylabel('Yüzdelik (%)')
    plt.title('Film Tema Dağılımı')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', rotation=0)

    return plt


def read_file_with_auto_encoding(file_path):
    """Dosya karakter kodlamasını otomatik tespit ederek okur."""
    with open(file_path, 'rb') as f:
        raw_data = f.read()

    result = chardet.detect(raw_data)
    encoding = result['encoding']

    encodings_to_try = [
        encoding,
        'utf-8',
        'cp1254',
        'iso-8859-9',
        'latin-5',
        'iso-8859-1',
        'windows-1252'
    ]

    for enc in encodings_to_try:
        if enc is None:
            continue
        try:
            text = raw_data.decode(enc)
            print(f"{enc} kodlamasıyla başarıyla çözüldü.")
            return text
        except UnicodeDecodeError:
            continue

    raise ValueError("Dosya bilinen hiçbir karakter kodlamasıyla çözülemedi.")


def process_file(file):
    try:
        if hasattr(file, 'name'):
            file_content = read_file_with_auto_encoding(file.name)
        else:
            if isinstance(file, bytes):
                result = chardet.detect(file)
                encoding = result['encoding']
                try:
                    file_content = file.decode(encoding)
                except UnicodeDecodeError:
                    for enc in ['utf-8', 'cp1254', 'iso-8859-9', 'latin-5', 'iso-8859-1']:
                        try:
                            file_content = file.decode(enc)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise ValueError("Dosya hiçbir yaygın kodlamayla çözümlenemedi.")
            elif isinstance(file, str):
                file_content = file
            else:
                raise ValueError("Desteklenmeyen dosya formatı. Lütfen metin tabanlı bir SRT dosyası yükleyin.")

        processed_sentences = process_subtitle_file(file_content)

        if not processed_sentences:
            return "Dosyada geçerli bir metin bulunamadı. Lütfen dosya formatını kontrol edin.", None

        chunks = create_chunks(processed_sentences)
        theme_percentages = predict_theme_distribution(chunks)
        fig = plot_theme_distribution(theme_percentages)

        report = "🎬 Film Tema Analizi Sonuçları:\n\n"
        for theme, percentage in sorted(theme_percentages.items(), key=lambda x: x[1], reverse=True):
            report += f"• {theme.capitalize()}: %{percentage:.1f}\n"

        return report, fig
    except Exception as e:
        return f"Bir hata oluştu: {str(e)}", None


# Gradio arayüzü
with gr.Blocks(title="Film Tema Analizi") as app:
    gr.Markdown("# 🎬 Film Tema Analizi Aracı")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Altyazı Dosyası Yükle (.srt veya .txt)", file_types=[".srt", ".txt"])
            analyze_btn = gr.Button("Temaları Analiz Et", variant="primary")

        with gr.Column():
            result_text = gr.Textbox(label="Analiz Sonuçları", lines=12, interactive=False)

    chart_output = gr.Plot(label="Tema Dağılımı Grafiği")

    analyze_btn.click(
        fn=process_file,
        inputs=[file_input],
        outputs=[result_text, chart_output]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()