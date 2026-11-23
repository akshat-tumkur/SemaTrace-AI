-- Apply once in Supabase SQL Editor, or with scripts/cloud_admin.py migrate.
-- Application credentials remain server-side. Browser roles have read-only RLS.
CREATE TABLE IF NOT EXISTS public.st_sources (
    owner_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    source_id text NOT NULL, metadata jsonb NOT NULL, active_version text,
    PRIMARY KEY(owner_id, source_id)
);
CREATE TABLE IF NOT EXISTS public.st_versions (
    owner_id uuid NOT NULL, version_id text NOT NULL, source_id text NOT NULL,
    metadata jsonb NOT NULL, raw_sha256 text NOT NULL, text_sha256 text NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(owner_id, version_id),
    FOREIGN KEY(owner_id,source_id) REFERENCES public.st_sources ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS public.st_passages (
    owner_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    passage_id text NOT NULL, text text NOT NULL, token_count integer NOT NULL,
    PRIMARY KEY(owner_id, passage_id)
);
CREATE TABLE IF NOT EXISTS public.st_occurrences (
    owner_id uuid NOT NULL, occurrence_id text NOT NULL, version_id text NOT NULL,
    passage_id text NOT NULL, source_id text NOT NULL, url text NOT NULL,
    start_char integer NOT NULL CHECK(start_char>=0), end_char integer NOT NULL,
    indexed boolean NOT NULL DEFAULT false,
    PRIMARY KEY(owner_id, occurrence_id), CHECK(end_char>start_char),
    FOREIGN KEY(owner_id,version_id) REFERENCES public.st_versions ON DELETE CASCADE,
    FOREIGN KEY(owner_id,passage_id) REFERENCES public.st_passages ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS st_occurrences_version ON public.st_occurrences(owner_id,version_id);
CREATE INDEX IF NOT EXISTS st_occurrences_passage ON public.st_occurrences(owner_id,passage_id);
CREATE TABLE IF NOT EXISTS public.st_terms (
    owner_id uuid NOT NULL, passage_id text NOT NULL, term text NOT NULL,
    frequency integer NOT NULL CHECK(frequency>0),
    PRIMARY KEY(owner_id,term,passage_id),
    FOREIGN KEY(owner_id,passage_id) REFERENCES public.st_passages ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS public.st_index_state (
    owner_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    space text NOT NULL, host_hash text NOT NULL
);
CREATE TABLE IF NOT EXISTS public.st_reports (
    owner_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    analysis_id text NOT NULL, payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(owner_id,analysis_id)
);
ALTER TABLE public.st_reports ADD COLUMN IF NOT EXISTS filename text NOT NULL DEFAULT '';
ALTER TABLE public.st_reports ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'completed';
UPDATE public.st_reports SET filename=coalesce(payload->>'filename','') WHERE filename='';
CREATE INDEX IF NOT EXISTS st_reports_created ON public.st_reports(owner_id,created_at DESC,analysis_id);
CREATE TABLE IF NOT EXISTS public.st_jobs (
    owner_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    job_id text NOT NULL, payload jsonb NOT NULL,
    PRIMARY KEY(owner_id,job_id)
);
CREATE TABLE IF NOT EXISTS public.st_usage (
    subject text NOT NULL, day date NOT NULL DEFAULT current_date,
    operation text NOT NULL, used integer NOT NULL,
    PRIMARY KEY(subject,day,operation)
);
CREATE TABLE IF NOT EXISTS public.st_vector_deletions (
    owner_id uuid NOT NULL, source_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(owner_id,source_id),
    FOREIGN KEY(owner_id,source_id) REFERENCES public.st_sources ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS public.st_snapshots (
    owner_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    version_id text NOT NULL, source_versions jsonb NOT NULL,
    PRIMARY KEY(owner_id,version_id)
);
DO $$
DECLARE tab text;
BEGIN
  FOREACH tab IN ARRAY ARRAY['st_sources','st_versions','st_passages','st_occurrences',
    'st_terms','st_index_state','st_reports','st_jobs','st_vector_deletions','st_snapshots']
  LOOP
    EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',tab);
    EXECUTE format('REVOKE ALL ON public.%I FROM anon, authenticated',tab);
    EXECUTE format('GRANT SELECT ON public.%I TO authenticated',tab);
    IF NOT EXISTS(SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename=tab AND policyname='owner_read') THEN
      EXECUTE format('CREATE POLICY owner_read ON public.%I FOR SELECT TO authenticated USING(owner_id=auth.uid())',tab);
    END IF;
  END LOOP;
END $$;
ALTER TABLE public.st_usage ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.st_usage FROM anon, authenticated;
