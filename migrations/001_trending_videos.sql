-- Migration 001: trending_videos and scraper_runs tables
-- Run this in Supabase SQL Editor

-- ============================================================
-- TABLE: trending_videos
-- Stores top-performing cosmetics ads from TikTok Creative Center
-- ============================================================
CREATE TABLE IF NOT EXISTS trending_videos (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  week_of       DATE NOT NULL,
  source        TEXT NOT NULL DEFAULT 'tiktok_creative_center',
  category      TEXT NOT NULL DEFAULT 'cosmetics',
  rank          INTEGER,
  ad_id         TEXT UNIQUE,
  ad_url        TEXT,
  caption       TEXT,
  brand_name    TEXT,
  objective     TEXT,
  region        TEXT DEFAULT 'BR',
  landing_page  TEXT,
  duration_s    INTEGER,
  likes         TEXT,
  comments      TEXT,
  shares        TEXT,
  ctr_level     TEXT,
  budget_level  TEXT,
  video_url     TEXT,
  cover_url     TEXT,
  raw_data      JSONB,
  ai_analysis   JSONB,
  is_active     BOOLEAN NOT NULL DEFAULT true,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trending_videos_week ON trending_videos(week_of);
CREATE INDEX IF NOT EXISTS idx_trending_videos_active ON trending_videos(is_active);
CREATE INDEX IF NOT EXISTS idx_trending_videos_ad_id ON trending_videos(ad_id);

-- ============================================================
-- TABLE: scraper_runs
-- Logs every scraper execution for monitoring
-- ============================================================
CREATE TABLE IF NOT EXISTS scraper_runs (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  finished_at   TIMESTAMPTZ,
  status        TEXT NOT NULL DEFAULT 'running',
  items_scraped INTEGER DEFAULT 0,
  error_message TEXT,
  week_of       DATE,
  source        TEXT DEFAULT 'tiktok_creative_center'
);

-- ============================================================
-- RLS: Allow service role full access (no public access)
-- ============================================================
ALTER TABLE trending_videos ENABLE ROW LEVEL SECURITY;
ALTER TABLE scraper_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access - trending_videos"
  ON trending_videos FOR ALL
  USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access - scraper_runs"
  ON scraper_runs FOR ALL
  USING (auth.role() = 'service_role');

