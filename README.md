# YouTube Comment Sentiment Analyzer 🎬📊

Расширение Chrome + Flask API, которое анализирует тональность комментариев под видео YouTube. Использует машинное обучение (LightGBM, TF-IDF) и NLP для определения позитивных, негативных и нейтральных отзывов. Всё упаковано в Docker и готово к локальному деплою.

## ✨ Возможности

- 🔍 Получение до 500 комментариев к любому видео через YouTube Data API v3.
- 🧠 Классификация тональности (`Positive`, `Neutral`, `Negative`) с помощью обученной модели LightGBM.
- 📈 Визуализация результатов:
  - Круговая диаграмма распределения сентимента
  - График трендов сентимента по месяцам
  - Облако слов
  - Сводная статистика (кол-во комментариев, уникальных авторов, средняя длина, средний сентимент)
- 🐳 Полностью контейнеризованное приложение (Docker).
- 🌐 Простое REST API (Flask).

## 🧱 Архитектура

Пользователь (YouTube) → Расширение Chrome (popup.js)
↓
HTTP REST API (Flask) ← Docker-контейнер
↓
ML-модель (LightGBM + TF-IDF) + NLP (NLTK)
text


- **Frontend**: Chrome Extension (HTML, CSS, JavaScript)
- **Backend**: Flask (Python)
- **ML**: scikit-learn, LightGBM, NLTK
- **Контейнеризация**: Docker, Docker Hub

## 📦 Стек технологий

| Категория       | Инструменты                                                                                         |
|-----------------|-----------------------------------------------------------------------------------------------------|
| Язык            | Python 3.11                                                                                         |
| Web-фреймворк   | Flask, Flask-CORS                                                                                   |
| Машинное обучение | scikit-learn 1.6.1, LightGBM, NLTK (stopwords, wordnet), TF-IDF                                  |
| Данные          | Pandas, NumPy                                                                                       |
| Визуализация    | Matplotlib, WordCloud                                                                               |
| Деплой          | Docker, Docker Hub, GitHub Actions (CI/CD)                                                          |
| Расширение      | Chrome Extensions Manifest V3                                                                       |

## 🚀 Быстрый старт (локальный запуск через Docker)

### Предварительные требования
- Установленный [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / Mac / Linux)
- Ключ YouTube Data API v3 ([инструкция](#-как-получить-youtube-api-key))

### 1. Скачайте образ с Docker Hub (рекомендуемый способ)
```bash
docker pull ntsundere/youtube:latest
docker run -d -p 5000:5000 ntsundere/youtube:latest

2. Проверьте, что сервер работает

Откройте в браузере:
text

http://localhost:5000/

Ожидаемый ответ: "Welcome to our flask api".
3. Установите расширение Chrome

    Скачайте файлы расширения из папки extension/ репозитория.

    Перейдите на chrome://extensions/.

    Включите режим разработчика (справа вверху).

    Нажмите Загрузить распакованное расширение и выберите папку extension.

    Убедитесь, что в popup.js значение API_URL установлено на http://localhost:5000/.

4. Вставьте свой YouTube API Key

В файле extension/popup.js замените значение API_KEY на ваш ключ:
javascript

const API_KEY = 'ВАШ_КЛЮЧ_ЮТУБ';

5. Запустите анализ

    Откройте любое видео на YouTube.

    Нажмите на иконку расширения.

    Наблюдайте за анализом комментариев 🚀.

🔧 Локальная разработка и сборка (если хотите изменить код)
Установка окружения
bash

# Клонируйте репозиторий
git clone <ваш-репозиторий>
cd YouTube-Comments-Checker

# Создайте виртуальное окружение
python -m venv youtube_env
youtube_env\Scripts\activate  # Windows
# source youtube_env/bin/activate  # Mac/Linux

# Установите зависимости
pip install -r requirements.txt

Сборка Docker-образа вручную
bash

docker build -t ntsundere/youtube .
docker run -d -p 5000:5000 ntsundere/youtube

Загрузка собственного образа в Docker Hub
bash

docker login -u ntsundere
docker tag ntsundere/youtube ntsundere/youtube:latest
docker push ntsundere/youtube:latest

📬 API Endpoints
Метод	URL	Тело запроса (JSON)	Ответ
GET	/	—	"Welcome to our flask api"
POST	/predict	{"comments": ["text1", "text2"]}	[{"comment": "text", "sentiment": 1}]
POST	/predict_with_timestamps	{"comments": [{"text": "..", "timestamp": ".."}]}	как выше + timestamp
POST	/generate_chart	{"sentiment_counts": {"1": 10, "0": 5, "-1": 3}}	PNG-изображение (pie chart)
POST	/generate_wordcloud	{"comments": ["word1 word2", ...]}	PNG-изображение (wordcloud)
POST	/generate_trend_graph	{"sentiment_data": [{"timestamp": "..", "sentiment": 1}, ...]}	PNG-изображение (тренд)
Пример запроса через curl
bash

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"comments": ["This video is awesome!", "Very bad explanation."]}'

Ответ:
json

[
  {"comment": "This video is awesome!", "sentiment": 1},
  {"comment": "Very bad explanation.", "sentiment": -1}
]

🔑 Как получить YouTube API Key

    Перейдите в Google Cloud Console.

    Создайте новый проект (или выберите существующий).

    Перейдите в раздел APIs & Services → Library.

    Найдите и включите YouTube Data API v3.

    В разделе Credentials нажмите Create Credentials → API Key.

    Скопируйте ключ и вставьте в popup.js.

    (Рекомендуется) Ограничьте ключ по HTTP-рефереру (для безопасности).

🧪 Тестирование API

Можно быстро проверить работоспособность прямо из терминала (PowerShell):
powershell

Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType 'application/json' -Body '{"comments": ["This is amazing!", "I hate it."]}'

📂 Структура проекта
text

YouTube-Comments-Checker/
├── app/
│   └── app.py                # Flask-приложение
├── extension/
│   ├── popup.html
│   ├── popup.js
│   └── manifest.json         # расширение Chrome
├── models/
│   ├── lgbm_model.pkl
│   └── tfidf_vectorizer.pkl
├── Dockerfile
├── requirements.txt
└── README.md

🔗 Полезные ссылки

    Docker Hub: https://hub.docker.com/r/ntsundere/youtube

    Видеоинструкция по получению API ключа: YouTube API Key Tutorial

📝 Лицензия

Проект создан в учебных целях. Свободное использование.