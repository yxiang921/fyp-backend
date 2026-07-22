-- Run once after upgrading from the original Python schema.
-- It moves Python-only profile data away from Laravel's missing-person tables.
CREATE TABLE IF NOT EXISTS face_profiles (
  external_id VARCHAR(100) PRIMARY KEY, full_name VARCHAR(150) NOT NULL,
  age SMALLINT, gender VARCHAR(50), notes TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
  -- The first version of this project used missing_people for Python profiles.
  -- Copy those records only when that legacy table exists with external_id.
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'missing_people' AND column_name = 'external_id') THEN
    EXECUTE 'INSERT INTO face_profiles (external_id, full_name, age, gender, notes, metadata)
             SELECT external_id, full_name, age, gender, notes, metadata FROM missing_people
             ON CONFLICT (external_id) DO NOTHING';
  END IF;
END $$;

ALTER TABLE face_embeddings DROP CONSTRAINT IF EXISTS face_embeddings_external_id_fkey;
ALTER TABLE face_embeddings
  ADD CONSTRAINT face_embeddings_external_id_fkey
  FOREIGN KEY (external_id) REFERENCES face_profiles(external_id) ON DELETE CASCADE;
