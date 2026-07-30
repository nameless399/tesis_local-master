BEGIN;

-- =========================================================
-- 1) ROLES
-- =========================================================
CREATE TABLE IF NOT EXISTS roles (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE
);

INSERT INTO roles (name)
VALUES ('operador'), ('jefe')
ON CONFLICT (name) DO NOTHING;


-- =========================================================
-- 2) USERS  (charge -> role_id)
-- =========================================================
ALTER TABLE users ADD COLUMN IF NOT EXISTS role_id INTEGER;
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'users'
          AND column_name = 'charge'
    ) THEN
        UPDATE users u
        SET role_id = r.id
        FROM roles r
        WHERE u.role_id IS NULL
          AND r.name = LOWER(COALESCE(u.charge, 'operador'));
    END IF;
END $$;

UPDATE users u
SET role_id = r.id
FROM roles r
WHERE u.role_id IS NULL
  AND r.name = 'operador';

ALTER TABLE users ALTER COLUMN role_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_users_role_id'
    ) THEN
        ALTER TABLE users
        ADD CONSTRAINT fk_users_role_id
        FOREIGN KEY (role_id) REFERENCES roles(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_users_role_id ON users(role_id);

ALTER TABLE users DROP COLUMN IF EXISTS charge;


-- =========================================================
-- 3) CAMERAS  (cam_id -> serial_number)
-- =========================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'cameras'
          AND column_name = 'cam_id'
    )
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'cameras'
          AND column_name = 'serial_number'
    ) THEN
        ALTER TABLE cameras RENAME COLUMN cam_id TO serial_number;
    END IF;
END $$;

ALTER TABLE cameras ADD COLUMN IF NOT EXISTS serial_number VARCHAR(100);
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS location_description VARCHAR(255) NOT NULL DEFAULT '';
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;

UPDATE cameras
SET serial_number = CONCAT('cam-', id)
WHERE serial_number IS NULL OR TRIM(serial_number) = '';

ALTER TABLE cameras ALTER COLUMN serial_number SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_cameras_serial_number'
    ) THEN
        ALTER TABLE cameras
        ADD CONSTRAINT uq_cameras_serial_number UNIQUE (serial_number);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_cameras_serial_number ON cameras(serial_number);


-- =========================================================
-- 4) CAMERA_ACTIONS  (cam_id -> camera_id FK)
-- =========================================================
ALTER TABLE camera_actions ADD COLUMN IF NOT EXISTS camera_id INTEGER;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'camera_actions'
          AND column_name = 'cam_id'
    ) THEN
        UPDATE camera_actions ca
        SET camera_id = c.id
        FROM cameras c
        WHERE ca.camera_id IS NULL
          AND ca.cam_id = c.serial_number;
    END IF;
END $$;

DELETE FROM camera_actions
WHERE camera_id IS NULL;

ALTER TABLE camera_actions ALTER COLUMN camera_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_camera_actions_camera_id'
    ) THEN
        ALTER TABLE camera_actions
        ADD CONSTRAINT fk_camera_actions_camera_id
        FOREIGN KEY (camera_id) REFERENCES cameras(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_camera_actions_camera_id ON camera_actions(camera_id);
ALTER TABLE camera_actions DROP COLUMN IF EXISTS cam_id;


-- =========================================================
-- 5) ACCESS_LOGS  (asegurar FK user_id)
-- =========================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_access_logs_user_id'
    ) THEN
        ALTER TABLE access_logs
        ADD CONSTRAINT fk_access_logs_user_id
        FOREIGN KEY (user_id) REFERENCES users(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_access_logs_user_id ON access_logs(user_id);


-- =========================================================
-- 6) ALERT_LOGS  (cam_id -> camera_id + validacion humana)
-- =========================================================
ALTER TABLE alert_logs ADD COLUMN IF NOT EXISTS camera_id INTEGER;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'alert_logs'
          AND column_name = 'cam_id'
    ) THEN
        UPDATE alert_logs al
        SET camera_id = c.id
        FROM cameras c
        WHERE al.camera_id IS NULL
          AND al.cam_id = c.serial_number;
    END IF;
END $$;

DELETE FROM alert_logs
WHERE camera_id IS NULL;

ALTER TABLE alert_logs ALTER COLUMN camera_id SET NOT NULL;
ALTER TABLE alert_logs ADD COLUMN IF NOT EXISTS status VARCHAR(30) NOT NULL DEFAULT 'pending';
ALTER TABLE alert_logs ADD COLUMN IF NOT EXISTS evidence_path VARCHAR(255);
ALTER TABLE alert_logs ADD COLUMN IF NOT EXISTS reviewed_by INTEGER;
ALTER TABLE alert_logs ADD COLUMN IF NOT EXISTS review_timestamp TIMESTAMP;

UPDATE alert_logs
SET status = 'pending'
WHERE status IS NULL OR status NOT IN ('pending', 'true_positive', 'false_positive');

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_alert_logs_status'
    ) THEN
        ALTER TABLE alert_logs
        ADD CONSTRAINT chk_alert_logs_status
        CHECK (status IN ('pending', 'true_positive', 'false_positive'));
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_alert_logs_camera_id'
    ) THEN
        ALTER TABLE alert_logs
        ADD CONSTRAINT fk_alert_logs_camera_id
        FOREIGN KEY (camera_id) REFERENCES cameras(id);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_alert_logs_reviewed_by'
    ) THEN
        ALTER TABLE alert_logs
        ADD CONSTRAINT fk_alert_logs_reviewed_by
        FOREIGN KEY (reviewed_by) REFERENCES users(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_alert_logs_camera_id ON alert_logs(camera_id);
CREATE INDEX IF NOT EXISTS ix_alert_logs_status ON alert_logs(status);
CREATE INDEX IF NOT EXISTS ix_alert_logs_reviewed_by ON alert_logs(reviewed_by);

ALTER TABLE alert_logs DROP COLUMN IF EXISTS cam_id;

COMMIT;
