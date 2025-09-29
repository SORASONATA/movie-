import pickle, pandas as pd
movies, cosine_sim = pickle.load(open('movie_data.pkl','rb'))
print(movies.head())                  # ดู 5 แถวแรก
print("\ncolumns:", list(movies.columns))
print("cosine_sim shape:", getattr(cosine_sim, "shape", None))
