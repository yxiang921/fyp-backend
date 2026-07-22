import os
import cv2
import pickle

from insightface.app import FaceAnalysis


IMAGE_FOLDER="images"

DB_FILE="database/face_db.pkl"



app = FaceAnalysis(
    name="buffalo_s"
)

app.prepare(
    ctx_id=-1,
    det_size=(320,320)
)



database={}



for file in os.listdir(IMAGE_FOLDER):

    if file.endswith(".jpg"):


        path=os.path.join(
            IMAGE_FOLDER,
            file
        )


        img=cv2.imread(path)


        faces=app.get(img)


        if len(faces)!=1:
            print(
                "Skip:",
                file
            )
            continue



        name=file.replace(
            ".jpg",
            ""
        )


        database[name]=faces[0].embedding


        print(
            "Registered:",
            name
        )



os.makedirs(
    "database",
    exist_ok=True
)



with open(
    DB_FILE,
    "wb"
) as f:

    pickle.dump(
        database,
        f
    )


print("Database Created")