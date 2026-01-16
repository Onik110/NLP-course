import os
from mistralai import Mistral
import pandas as pd
import time
import json
from dotenv import load_dotenv

df = pd.read_csv("dataset_jug.csv")

# Добавление talk_id: от 0 до N-1
df = df.reset_index(drop=True)  
df['talk_id'] = df.index  

print("Столбец 'talk_id' успешно добавлен.")
print(df[['talk_id', 'title']].head())

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") # задается через .env
MODEL = "mistral-large-latest" 

client = Mistral(api_key=MISTRAL_API_KEY)

def get_similarity_score(title1, text1, title2, text2, retries=3):
    prompt = f"""You are an expert in mobile development, QA, and software engineering conferences.
Given two conference talk descriptions below, rate their thematic similarity on a scale from 0.0 (completely unrelated) to 1.0 (identical topic and focus).

Talk 1:
Title: "{title1}"
Description: "{text1}"

Talk 2:
Title: "{title2}"
Description: "{text2}"

Respond ONLY with a number between 0.0 and 1.0, rounded to one decimal place."""

    for attempt in range(retries):
        try:
            response = client.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            raw = response.choices[0].message.content.strip()
            score = float(raw)
            if 0.0 <= score <= 1.0:
                return round(score, 1)
            else:
                print(f"Invalid score: {raw}")
                return 0.0
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(2)
    return 0.0 


cache_file = "similarity_cache.json"

# Загрузка кэша
if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

total_pairs = 0
new_pairs = 0

N = len(df)  # полная матрица

for i in range(N):
    for j in range(i + 1, N):
        total_pairs += 1
        key = f"{df.iloc[i]['talk_id']}_{df.iloc[j]['talk_id']}"
        
        if key in cache:
            continue  
        
        new_pairs += 1
        row1 = df.iloc[i]
        row2 = df.iloc[j]
        
        score = get_similarity_score(
            row1["title"], 
            row1["text"] if pd.notna(row1["text"]) else "",
            row2["title"], 
            row2["text"] if pd.notna(row2["text"]) else ""
        )
        
        cache[key] = float(score)
        print(f"[{row1['talk_id']}, {row2['talk_id']}] → {score}")
        
        # Сохраняем кэш после каждого вызова 
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        
        time.sleep(0.5)  # rate limits Mistral API

print(f"Обработано: {new_pairs} новых пар из {total_pairs} возможных.")
print(f"Кэш сохранён в '{cache_file}'")