ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS expected_speakers INT;

ALTER TABLE sessions
    DROP CONSTRAINT IF EXISTS sessions_expected_speakers_check;

ALTER TABLE sessions
    ADD CONSTRAINT sessions_expected_speakers_check
        CHECK (expected_speakers IS NULL OR expected_speakers BETWEEN 1 AND 20);
