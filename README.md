# Missing Person Recognition (Python)

This Flask application implements the recognition pipeline:

`camera/upload → YOLO person detection → ByteTrack/BoT-SORT → SCRFD face detection → ArcFace comparison → Laravel alert`

## Run

Open **Anaconda Prompt** (not a normal PowerShell window), then run:

```powershell
conda activate missing_person_backend
cd "E:\Southern University College\FYP\missing_person_backend"
pip install -r requirements.txt
python main.py
```

Open `http://localhost:5000`.

`/video_feed` is an MJPEG endpoint for embedding in Laravel, for example:

```html
<img src="http://localhost:5000/video_feed" alt="Missing person camera">
```

## Web functions

- **Register missing person**: stores one ArcFace embedding and name, age, gender, and notes in `database/`.
- **Image testing**: uploads a still image and displays annotated persons/faces and recognition scores.
- **Video testing**: uploads a video and produces an annotated MP4 in `outputs/`.
- **Live laptop camera**: recognizes each tracked person and exposes the annotated feed through `/video_feed`.

## Settings

The default similarity threshold is `0.50` in `face_service.py`. Recognized targets trigger a blank-body `POST` to `http://localhost:8000/sendAlert`, with a 30-second per-person/per-track cooldown to avoid duplicate alerts.

To use BoT-SORT rather than ByteTrack for one run:

```powershell
$env:TRACKER="botsort.yaml"
python main.py
```

Optional environment settings: `CAMERA_INDEX` (default `0`), `PORT` (default `5000`), and `FLASK_SECRET_KEY`.

## Video performance

The app now runs person tracking every frame but runs SCRFD + ArcFace only once every four frames for each existing track. This is normally the largest speed improvement while keeping boxes smooth. Large videos are also resized to a maximum inference width of 960 pixels and restored to their original output size.

For a faster, lower-detail demo, set these before starting the app:

```powershell
$env:FACE_INTERVAL="8"         # ArcFace/SCRFD once every 8 frames per tracked person
$env:INFERENCE_MAX_WIDTH="640" # smaller YOLO/face-analysis input
python main.py
```

For best speed, use `yolov8n.pt`, a CUDA-enabled PyTorch/ONNX Runtime installation, a 640p source clip, and short videos. CPU-only inference will be substantially slower.

## Notes

On the first run, Ultralytics and InsightFace may download their required model weights. Use a clear registration photo with exactly one visible face. The database format remains compatible with the previous `database/face_db.pkl` file.

## PostgreSQL and Laravel registration API

1. Copy `.env.example` to a private `.env` file or set its values in your deployment environment. Do **not** commit database credentials.
2. Run `schema.sql` once against PostgreSQL.
   If you previously used this project's old PostgreSQL schema, also run `migrations/001_move_face_embeddings_to_face_profiles.sql` once.
3. Install the updated dependencies and set `DATABASE_URL`. The Python app automatically uses PostgreSQL when that variable exists; otherwise it continues with the local demonstration database.

Laravel registers a person with `POST /api/missing-persons` as `multipart/form-data`:

```bash
curl -X POST http://localhost:5000/api/missing-persons \
  -H "X-API-Key: YOUR_LARAVEL_API_KEY" \
  -F "case_number=MP-2026-001" -F "fullname=Jane Doe" \
  -F "photo=@/path/to/jane.jpg"
```

The endpoint accepts only `fullname`, `case_number`, and `photo`; it requires exactly one face and returns `201` with `status: registered`. It returns `422` when the photo contains zero or multiple faces. Set `LARAVEL_API_KEY` so unauthenticated requests are rejected.

The received `case_number` is stored with the ArcFace profile. When a camera recognizes that profile, Python saves an annotated frame under `outputs/detections/`, inserts a `detections` row, then creates a pending `verification_requests` row. Set these environment variables to connect that flow to Laravel:

```env
MISSING_PERSON_TABLE=missing_persons
PYTHON_SYSTEM_USER_ID=1
```

`MISSING_PERSON_TABLE` must be Laravel's table containing `id` and `case_number`; for the standard `MissingPerson` convention use `missing_persons`. `PYTHON_SYSTEM_USER_ID` is the Laravel user ID recorded as `verification_requests.requested_by`.

## Multiple CCTV streams

When PostgreSQL is configured, cameras are read from Laravel's shared `cameras` table (only rows with `status = 'online'`). Cameras can be created from either Laravel or the Flask Settings page; Flask's form inserts into this same table and requires the Laravel `registered_by` user ID. The source uses `stream_url` first; if it is empty, Flask falls back to `http://ip_address:port`.

Without PostgreSQL, local camera definitions live in `database/settings.json`. Example:

```json
{"id":"entrance","name":"Main entrance","source":"rtsp://user:password@camera-host:554/stream1","enabled":true}
```

You can also add or remove these cameras from the Flask **Settings** page. Added cameras remain stopped until **Start recognition** is selected on the dashboard.

Each active camera is available at `/video_feed/<id>` and has its own YOLO tracker and ArcFace cache, preventing track IDs from different cameras from interfering. For a real deployment, run one worker/process per camera or use a queue-based worker service, use GPU-backed ONNX Runtime/PyTorch, keep PostgreSQL on a private network, and put the Flask UI behind an authenticated reverse proxy.
