# enrich_director.py
import pickle
import requests
import pandas as pd
from tqdm import tqdm

# ✅ ใส่ API Key ของคุณตรงนี้แล้ว
TMDB_API_KEY = "b964b8682fdd7c07019be4031b66f77c"

in_path = "movie_data_with_director.pkl"
out_path = "movie_data_with_director_filled.pkl"

# โหลดข้อมูลเดิม
with open(in_path, "rb") as f:
    movies, cosine_sim = pickle.load(f)

def get_director(movie_id):
    """ดึงชื่อผู้กำกับจาก TMDB API ตาม movie_id"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
    try:
        res = requests.get(url)
        if res.status_code != 200:
            return ""
        data = res.json()
        crew = data.get("crew", [])
        for member in crew:
            if member.get("job") == "Director":
                return member.get("name", "")
    except Exception:
        return ""
    return ""

# เติมข้อมูลลงในคอลัมน์ director (ถ้าว่าง)
if "director" not in movies.columns:
    movies["director"] = ""

print("กำลังดึงข้อมูลผู้กำกับจาก TMDB...")
for i in tqdm(range(len(movies))):
    if movies.at[i, "director"] == "":  # เติมเฉพาะที่ยังว่าง
        movie_id = movies.at[i, "movie_id"]
        director = get_director(movie_id)
        movies.at[i, "director"] = director

# เซฟเป็นไฟล์ใหม่
with open(out_path, "wb") as f:
    pickle.dump((movies, cosine_sim), f)

print(f"✅ บันทึกเรียบร้อย: {out_path}")
