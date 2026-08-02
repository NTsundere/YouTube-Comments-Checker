import numpy as np
import pandas as pd
import pickle
import os
import time
import onnxruntime as ort
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(ROOT_DIR, 'lgbm_model.pkl')
VECTORIZER_PATH = os.path.join(ROOT_DIR, 'tfidf_vectorizer.pkl')
ONNX_PATH = os.path.join(ROOT_DIR, 'lgbm_model.onnx')

print("Loading model and vectorizer...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

num_features = len(vectorizer.get_feature_names_out())
print(f"Number of features: {num_features}")

if not os.path.exists(ONNX_PATH):
    print("Converting model to ONNX...")
    initial_type = [('input', FloatTensorType([None, num_features]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type)
    with open(ONNX_PATH, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model saved to {ONNX_PATH}")
else:
    print("ONNX model already exists, skipping conversion.")

test_data_path = os.path.join(ROOT_DIR, 'data', 'interim', 'test_processed.csv')
if os.path.exists(test_data_path):
    test_df = pd.read_csv(test_data_path)
    sample_texts = test_df['clean_comment'].head(10).tolist()  # 10 примеров
else:
    sample_texts = [
        "I love this product, it's amazing!",
        "Terrible service, will never buy again.",
        "It's okay, not the best but not the worst.",
        "Absolutely fantastic experience!",
        "Waste of money, totally disappointed."
    ]

print("Transforming texts to TF-IDF...")
X_tfidf_sparse = vectorizer.transform(sample_texts)
X_tfidf_dense = X_tfidf_sparse.toarray().astype(np.float32)  # ONNX ожидает float32

print("\nPredictions with original LightGBM model:")
start_orig = time.perf_counter()
y_pred_orig = model.predict(X_tfidf_dense)
time_orig = time.perf_counter() - start_orig
print(f"Predictions: {y_pred_orig}")
print(f"Time for {len(sample_texts)} samples: {time_orig:.6f} sec")

print("\nLoading ONNX model and running inference...")
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

start_onnx = time.perf_counter()
outputs = sess.run(None, {input_name: X_tfidf_dense})
time_onnx = time.perf_counter() - start_onnx
y_pred_onnx = outputs[0]  
print(f"Predictions: {y_pred_onnx}")
print(f"Time for {len(sample_texts)} samples: {time_onnx:.6f} sec")

print("\nComparison of predictions (original vs ONNX):")
for i, (orig, onnx) in enumerate(zip(y_pred_orig, y_pred_onnx)):
    print(f"Sample {i+1}: Original={orig}, ONNX={onnx}, Match={orig == onnx}")

if np.array_equal(y_pred_orig, y_pred_onnx):
    print("\n✅ All predictions match exactly!")
else:
    print("\n⚠️ Some predictions differ (floating point rounding might cause small differences).")

if os.path.exists(test_data_path):
    large_sample = test_df['clean_comment'].fillna('').head(2000).tolist()
    X_large = vectorizer.transform(large_sample).toarray().astype(np.float32)

    start = time.perf_counter()
    _ = model.predict(X_large)
    time_large_orig = time.perf_counter() - start

    start = time.perf_counter()
    _ = sess.run(None, {input_name: X_large})
    time_large_onnx = time.perf_counter() - start

    print(f"\nPerformance on {len(large_sample)} samples:")
    print(f"Original LightGBM: {time_large_orig:.4f} sec")
    print(f"ONNX Runtime:      {time_large_onnx:.4f} sec")
    print(f"Speedup:           {time_large_orig / time_large_onnx:.2f}x")
else:
    print("\nSkipping large-scale performance test (test_processed.csv not found).")