-- Additive/resumable migration. Existing private UUIDs and all evidence stay intact.
CREATE TABLE IF NOT EXISTS public.st_libraries (
  library_id uuid PRIMARY KEY, kind text NOT NULL CHECK(kind IN('global','private')),
  user_id uuid UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  CHECK((kind='global' AND library_id='00000000-0000-0000-0000-000000000001'::uuid AND user_id IS NULL)
     OR (kind='private' AND user_id IS NOT NULL AND library_id=user_id))
);
INSERT INTO st_libraries VALUES('00000000-0000-0000-0000-000000000001','global',NULL) ON CONFLICT DO NOTHING;
INSERT INTO st_libraries SELECT id,'private',id FROM auth.users ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS public.st_admins (
  user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  granted_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE st_admins ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON st_admins FROM PUBLIC,anon,authenticated;
ALTER TABLE st_libraries ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON st_libraries FROM PUBLIC,anon,authenticated;
GRANT SELECT ON st_libraries TO authenticated;
DROP POLICY IF EXISTS library_read ON st_libraries;
CREATE POLICY library_read ON st_libraries FOR SELECT TO authenticated
  USING(auth.uid() IS NOT NULL AND (kind='global' OR user_id=auth.uid()));
DO $$ DECLARE tab text; fk record;
BEGIN
  FOREACH tab IN ARRAY ARRAY['st_sources','st_passages','st_index_state'] LOOP
    FOR fk IN SELECT conname FROM pg_constraint WHERE conrelid=('public.'||tab)::regclass
      AND contype='f' AND confrelid='auth.users'::regclass LOOP
      EXECUTE format('ALTER TABLE %I DROP CONSTRAINT %I',tab,fk.conname);
    END LOOP;
    IF NOT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid=('public.'||tab)::regclass AND conname=tab||'_library_fkey') THEN
      EXECUTE format('ALTER TABLE %I ADD CONSTRAINT %I FOREIGN KEY(owner_id) REFERENCES st_libraries(library_id) ON DELETE CASCADE',tab,tab||'_library_fkey');
    END IF;
  END LOOP;
  FOREACH tab IN ARRAY ARRAY['st_sources','st_versions','st_passages','st_occurrences','st_terms','st_index_state','st_vector_deletions'] LOOP
    EXECUTE format('DROP POLICY IF EXISTS owner_read ON %I',tab);
    EXECUTE format('CREATE POLICY owner_read ON %I FOR SELECT TO authenticated USING(auth.uid() IS NOT NULL AND (owner_id=auth.uid() OR owner_id=''00000000-0000-0000-0000-000000000001''::uuid))',tab);
  END LOOP;
END $$;
ALTER TABLE st_snapshots ADD COLUMN IF NOT EXISTS source_libraries jsonb NOT NULL DEFAULT '{}';
ALTER TABLE st_versions ADD COLUMN IF NOT EXISTS published_at timestamptz;
-- Query terms across eligible libraries without probing every passage/term pair.
CREATE INDEX IF NOT EXISTS st_terms_lookup ON st_terms(term,owner_id,passage_id) INCLUDE(frequency);
CREATE TABLE IF NOT EXISTS public.st_global_promotions (
  from_owner uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  source_id text NOT NULL, version_id text NOT NULL,
  state text NOT NULL DEFAULT 'copying', verified_vectors integer NOT NULL DEFAULT 0,
  vector_digest text, updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(from_owner,source_id,version_id)
);
ALTER TABLE st_global_promotions ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON st_global_promotions FROM PUBLIC,anon,authenticated;
DROP POLICY IF EXISTS sematrace_global_download ON storage.objects;
CREATE POLICY sematrace_global_download ON storage.objects FOR SELECT TO authenticated
  USING(auth.uid() IS NOT NULL AND bucket_id='sematrace-private'
    AND (storage.foldername(name))[1]='00000000-0000-0000-0000-000000000001'
    AND EXISTS(SELECT 1 FROM st_versions v WHERE v.owner_id='00000000-0000-0000-0000-000000000001'
      AND v.published_at IS NOT NULL AND split_part(name,'/',3) IN(v.raw_sha256,v.text_sha256)));
