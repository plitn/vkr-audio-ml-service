CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT NOT NULL DEFAULT 'pending',
    nr          BOOLEAN     NOT NULL DEFAULT false,
    asr         BOOLEAN     NOT NULL DEFAULT false,
    diar        BOOLEAN     NOT NULL DEFAULT false,
    audio_key TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'english',
    result JSONB NULL,
    error TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    finished_at TIMESTAMP NULL
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX jobs_status_index on jobs(status);
CREATE INDEX jobs_id_index on jobs(id);

ALTER TABLE jobs ADD COLUMN user_id UUID REFERENCES users(id);
CREATE INDEX idx_jobs_user_id ON jobs (user_id);