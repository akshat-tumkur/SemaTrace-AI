"""Optional hosted adapters; local operation does not require cloud credentials."""

import os


def enabled():
    return os.getenv("SEMATRACE_STORAGE_PROVIDER", "sqlite") == "supabase"
