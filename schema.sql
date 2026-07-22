CREATE TABLE IF NOT EXISTS face_profiles (
  external_id VARCHAR(100) PRIMARY KEY, full_name VARCHAR(150) NOT NULL,
  age SMALLINT, gender VARCHAR(50), notes TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS face_embeddings (
  external_id VARCHAR(100) PRIMARY KEY REFERENCES face_profiles(external_id) ON DELETE CASCADE,
  embedding BYTEA NOT NULL, dimensions SMALLINT NOT NULL DEFAULT 512,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS face_profiles_name_idx ON face_profiles(full_name);

-- These tables may instead be created by Laravel migrations with the same columns.
CREATE TABLE IF NOT EXISTS detections (
  id BIGSERIAL PRIMARY KEY, detection_id VARCHAR(50) NOT NULL UNIQUE,
  camera_id BIGINT NOT NULL, captured_image VARCHAR(255), reference_type VARCHAR(255) NOT NULL,
  reference_id BIGINT NOT NULL, similarity_score NUMERIC(5,2) NOT NULL DEFAULT 0,
  status VARCHAR(255) NOT NULL DEFAULT 'new', detected_at TIMESTAMP(0),
  reviewed_by BIGINT, review_notes TEXT, created_at TIMESTAMP(0), updated_at TIMESTAMP(0)
);
CREATE TABLE IF NOT EXISTS verification_requests (
  id BIGSERIAL PRIMARY KEY, request_id VARCHAR(50) NOT NULL UNIQUE,
  detection_id BIGINT NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
  alert_id BIGINT, requested_by BIGINT NOT NULL, reviewed_by BIGINT,
  status VARCHAR(255) NOT NULL DEFAULT 'pending', jpn_notes TEXT, police_notes TEXT,
  verified_at TIMESTAMP(0), created_at TIMESTAMP(0), updated_at TIMESTAMP(0)
);
CREATE INDEX IF NOT EXISTS detections_camera_detected_idx ON detections(camera_id, detected_at DESC);
