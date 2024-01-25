import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ["Miguel Retamozo", "German Ibarra"]
usernames = ["mretamozo", "gibarra"]
passwords = ["1qaz5thn", "1qaz6yjm"]


#hashed_passwords = stauth.Hasher(passwords).generate()
#
#file_path = Path(__file__).parent / "hashed_pw.pkl"
#with file_path.open("wb") as file:
#    pickle.dump(hashed_passwords, file)

hashed_passwords = stauth.Hasher(passwords).generate()
print(hashed_passwords)
