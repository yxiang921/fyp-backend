from flask import Flask, Response

import cv2

import main



app = Flask(__name__)



def generate_frames():


    while True:

        print(main.output_frame)
        if main.output_frame is None:

            print("Waiting frame...")
            continue



        frame = main.output_frame



        success,buffer=cv2.imencode(
            ".jpg",
            frame
        )


        if not success:

            continue



        frame_bytes=buffer.tobytes()



        yield(
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            +
            frame_bytes
            +
            b"\r\n"
        )




@app.route("/video_feed")

def video_feed():


    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )




@app.route("/")

def index():

    return """
    <html>

    <body>

    <h1>
    Missing Person CCTV
    </h1>


    <img src="/video_feed"
    width="800">


    </body>

    </html>
    """




if __name__=="__main__":


    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )