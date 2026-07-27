-- Run after full_schema.sql. Creates or updates the initial PDRM IT Super Admin.
-- The hash is Werkzeug PBKDF2-SHA256 and requires no PostgreSQL extension.

INSERT INTO users (name, email, password, role, is_active, created_at, updated_at)
VALUES (
    'PDRM IT Administrator',
    'admin@example.gov.my',
    'pbkdf2:sha256:1000000$frms_admin_2026$a9c97dcb5079a42314c715771c755548440f3fb1ae8cbc69391ebbfcb1e58dc0',
    'super_admin',
    TRUE,
    now(),
    now()
)
ON CONFLICT (email) DO UPDATE
SET name = EXCLUDED.name,
    password = EXCLUDED.password,
    role = 'super_admin',
    is_active = TRUE,
    updated_at = now();
