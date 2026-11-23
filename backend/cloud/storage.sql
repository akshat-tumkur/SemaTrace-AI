-- Supabase Storage private bucket. Browser writes are deliberately prohibited:
-- upload/ingestion limits must be enforced by the authenticated FastAPI backend.
INSERT INTO storage.buckets(id,name,public,file_size_limit)
VALUES('sematrace-private','sematrace-private',false,10485760)
ON CONFLICT(id) DO UPDATE SET public=false,file_size_limit=10485760;
DO $$ BEGIN
  IF NOT EXISTS(SELECT 1 FROM pg_policies WHERE schemaname='storage' AND tablename='objects' AND policyname='sematrace_owner_download') THEN
    CREATE POLICY sematrace_owner_download ON storage.objects
      FOR SELECT TO authenticated USING(
        bucket_id='sematrace-private' AND (storage.foldername(name))[1]=auth.uid()::text
      );
  END IF;
END $$;
