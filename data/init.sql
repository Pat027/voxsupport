-- voxsupport — synthetic Acme Cloud schema + seed data

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ==========================================================================
-- Accounts + billing
-- ==========================================================================

CREATE TABLE IF NOT EXISTS accounts (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email        TEXT UNIQUE NOT NULL,
    full_name    TEXT NOT NULL,
    plan         TEXT NOT NULL CHECK (plan IN ('starter', 'growth', 'scale', 'enterprise')),
    region       TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    status       TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'cancelled'))
);

CREATE TABLE IF NOT EXISTS bills (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id     UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    period_start   DATE NOT NULL,
    period_end     DATE NOT NULL,
    amount_cents   INTEGER NOT NULL,
    currency       TEXT NOT NULL DEFAULT 'EUR',
    status         TEXT NOT NULL CHECK (status IN ('open', 'paid', 'overdue')),
    issued_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tickets (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id     UUID REFERENCES accounts(id) ON DELETE SET NULL,
    subject        TEXT NOT NULL,
    priority       TEXT NOT NULL CHECK (priority IN ('low', 'normal', 'high', 'critical')),
    status         TEXT NOT NULL DEFAULT 'open',
    assigned_team  TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Long-term user preferences (per JD: "memory architecture")
CREATE TABLE IF NOT EXISTS user_preferences (
    account_id   UUID PRIMARY KEY REFERENCES accounts(id) ON DELETE CASCADE,
    language     TEXT NOT NULL DEFAULT 'en',
    voice_speed  REAL NOT NULL DEFAULT 1.0,
    last_topics  JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==========================================================================
-- Knowledge base — pgvector for RAG over support docs
-- ==========================================================================

CREATE TABLE IF NOT EXISTS kb_chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_slug     TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector(384),   -- all-MiniLM-L6-v2 dimension
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_slug, chunk_index)
);

CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx
    ON kb_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS kb_chunks_slug_idx ON kb_chunks (doc_slug);

-- ==========================================================================
-- Seed accounts (5 personas for demo + benchmarks)
-- ==========================================================================

INSERT INTO accounts (id, email, full_name, plan, region, status) VALUES
    ('11111111-1111-1111-1111-111111111111', 'alice@example.com',  'Alice Nguyen',    'growth',     'fra-1',    'active'),
    ('22222222-2222-2222-2222-222222222222', 'bob@example.com',    'Bob Schmidt',     'starter',    'fra-1',    'active'),
    ('33333333-3333-3333-3333-333333333333', 'carol@example.com',  'Carol Jansen',    'scale',      'ams-1',    'active'),
    ('44444444-4444-4444-4444-444444444444', 'dan@example.com',    'Dan Costa',       'enterprise', 'par-1',    'active'),
    ('55555555-5555-5555-5555-555555555555', 'eve@example.com',    'Eve Ostrowska',   'growth',     'waw-1',    'suspended')
ON CONFLICT (email) DO NOTHING;

INSERT INTO bills (account_id, period_start, period_end, amount_cents, currency, status) VALUES
    ('11111111-1111-1111-1111-111111111111', '2026-03-01', '2026-03-31',  4900, 'EUR', 'paid'),
    ('11111111-1111-1111-1111-111111111111', '2026-04-01', '2026-04-30',  4900, 'EUR', 'open'),
    ('22222222-2222-2222-2222-222222222222', '2026-03-01', '2026-03-31',   900, 'EUR', 'paid'),
    ('22222222-2222-2222-2222-222222222222', '2026-04-01', '2026-04-30',   900, 'EUR', 'open'),
    ('33333333-3333-3333-3333-333333333333', '2026-04-01', '2026-04-30', 12900, 'EUR', 'open'),
    ('44444444-4444-4444-4444-444444444444', '2026-04-01', '2026-04-30', 48900, 'EUR', 'paid'),
    ('55555555-5555-5555-5555-555555555555', '2026-03-01', '2026-03-31',  4900, 'EUR', 'overdue');

INSERT INTO user_preferences (account_id, language, voice_speed) VALUES
    ('11111111-1111-1111-1111-111111111111', 'en',  1.0),
    ('22222222-2222-2222-2222-222222222222', 'de',  1.0),
    ('33333333-3333-3333-3333-333333333333', 'nl',  1.1),
    ('44444444-4444-4444-4444-444444444444', 'fr',  0.95),
    ('55555555-5555-5555-5555-555555555555', 'pl',  1.0)
ON CONFLICT (account_id) DO NOTHING;
