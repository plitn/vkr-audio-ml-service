CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),

    status TEXT NOT NULL DEFAULT 'recording'
        CHECK (status IN ('recording', 'recording_finished', 'queued', 'processing', 'done', 'failed')),
    language TEXT NOT NULL DEFAULT 'auto',

    nr BOOLEAN NOT NULL DEFAULT true,
    asr BOOLEAN NOT NULL DEFAULT true,
    diar BOOLEAN NOT NULL DEFAULT true,
    diarization_mode TEXT NOT NULL DEFAULT 'full'
        CHECK (diarization_mode IN ('full', 'chunk')),

    chunk_duration_sec INT NOT NULL DEFAULT 30,
    total_duration_sec DOUBLE PRECISION,

    final_result JSONB,
    error TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    recording_started_at TIMESTAMPTZ,
    recording_finished_at TIMESTAMPTZ,

    processing_started_at TIMESTAMPTZ,
    processing_finished_at TIMESTAMPTZ
);


CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,

    chunk_index INT NOT NULL,
    status TEXT NOT NULL DEFAULT 'uploaded'
        CHECK (status IN ('uploaded', 'queued', 'processing', 'done', 'failed')),

    audio_key TEXT NOT NULL,
    normalized_audio_key TEXT,
    enhanced_audio_key TEXT,

    start_sec DOUBLE PRECISION,
    duration_sec DOUBLE PRECISION,
    is_final BOOLEAN NOT NULL DEFAULT false,

    result JSONB,
    error TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    processing_started_at TIMESTAMPTZ,
    processing_finished_at TIMESTAMPTZ,

    UNIQUE(session_id, chunk_index)
);


CREATE INDEX idx_chunks_session_id ON chunks (session_id);
CREATE INDEX idx_sessions_user_id  ON sessions (user_id);
