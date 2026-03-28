from datasets import load_dataset
import pandas as pd

print("Downloading the correct Emotion Dataset...")
# جلب البيانات من HuggingFace
dataset = load_dataset("dair-ai/emotion", "split")

# تحويلها إلى جدول وحفظها بصيغة CSV
df = pd.DataFrame(dataset['train'])

# الكود الخاص بك يحتاج عمود 'emotion' كنص
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
df['emotion'] = df['label'].apply(lambda x: emotion_labels[x])

df.to_csv("emotion_data.csv", index=False)
print("Done! File 'emotion_data.csv' is ready in your folder.")