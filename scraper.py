"""
TikTok Creative Center Scraper
Coleta os top ads de cosmeticos semanalmente via Playwright (route interception).
Exposto como endpoint POST /scrape-trending no worker FastAPI.
"""

import os
import asyncio
from datetime import date, timedelta
from typing import Optional

import httpx
from playwright.async_api import async_playwright

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

TIKTOK_CC_URL = (
    "https://ads.tiktok.com/business/creativecenter/inspiration/topads/pc/en"
    "?period=7&region=BR&secondIndustry=14104000000"
)
API_PATTERN = "creative_radar_api/v1/top_ads/v2/list"
MAX_ADS = 20


# ─────────────────────────────────────────────
# HELPERS: SUPABASE
# ─────────────────────────────────────────────
def supabase_headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


async def create_scraper_run(week_of: date) -> Optional[str]:
    url = f"{SUPABASE_URL}/rest/v1/scraper_runs"
    payload = {"week_of": str(week_of), "status": "running"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                url,
                headers={**supabase_headers(), "Prefer": "return=representation"},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            return data[0]["id"] if data else None
    except Exception as e:
        print(f"[scraper] Aviso: nao foi possivel criar scraper_run: {e}")
        return None


async def finish_scraper_run(
    run_id: Optional[str], status: str, items: int, error: str = None
):
    if not run_id:
        return
    url = f"{SUPABASE_URL}/rest/v1/scraper_runs?id=eq.{run_id}"
    payload = {"status": status, "items_scraped": items, "finished_at": "now()"}
    if error:
        payload["error_message"] = error[:500]
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.patch(url, headers=supabase_headers(), json=payload)
    except Exception as e:
        print(f"[scraper] Aviso: nao foi possivel atualizar scraper_run: {e}")


async def deactivate_old_ads(week_of: date):
    url = f"{SUPABASE_URL}/rest/v1/trending_videos?week_of=neq.{week_of}&is_active=eq.true"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.patch(url, headers=supabase_headers(), json={"is_active": False})
    except Exception as e:
        print(f"[scraper] Aviso ao desativar antigos: {e}")


async def upsert_trending_video(item: dict, week_of: date, rank: int):
    ad_id = str(item.get("id", ""))
    if not ad_id:
        return

    video_info = item.get("video_info", {}) or {}
    payload = {
        "week_of": str(week_of),
        "rank": rank,
        "ad_id": ad_id,
        "ad_url": (
            f"https://ads.tiktok.com/business/creativecenter/"
            f"topads/{ad_id}/pc/en?countryCode=BR&period=7"
        ),
        "caption": item.get("ad_title") or item.get("title") or "",
        "brand_name": item.get("brand_name") or item.get("advertiser_name") or "",
        "objective": item.get("objective_type") or item.get("objective") or "",
        "region": "BR",
        "landing_page": item.get("landing_page_url") or item.get("click_url") or "",
        "duration_s": int(video_info.get("duration", 0) or 0),
        "likes": str(item.get("like_count") or item.get("likes") or ""),
        "comments": str(item.get("comment_count") or item.get("comments") or ""),
        "shares": str(item.get("share_count") or item.get("shares") or ""),
        "ctr_level": item.get("ctr_level") or item.get("ctr_grade") or "",
        "budget_level": item.get("budget_level") or item.get("cost_level") or "",
        "video_url": video_info.get("url") or video_info.get("video_url") or "",
        "cover_url": video_info.get("cover") or video_info.get("thumbnail") or "",
        "raw_data": item,
        "is_active": True,
    }

    url = f"{SUPABASE_URL}/rest/v1/trending_videos"
    headers = {
        **supabase_headers(),
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, headers=headers, json=payload)
    except Exception as e:
        print(f"[scraper] Erro ao salvar ad {ad_id}: {e}")


# ─────────────────────────────────────────────
# PLAYWRIGHT SCRAPER
# ─────────────────────────────────────────────
async def scrape_tiktok_creative_center() -> list:
    """
    Abre TikTok Creative Center via Playwright e intercepta
    a resposta da API interna top_ads/v2/list para cosmeticos.
    """
    captured_materials = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--ignore-certificate-errors",
            ],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="pt-BR",
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()
        # Ocultar fingerprints de automacao
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['pt-BR', 'pt', 'en-US'] });
            window.chrome = { runtime: {}, loadTimes: function(){}, csi: function(){}, app: {} };
            Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
        """)

        async def handle_response(response):
            if API_PATTERN in response.url and "list" in response.url:
                try:
                    body = await response.json()
                    materials = (
                        body.get("data", {}).get("materials")
                        or body.get("data", {}).get("list")
                        or body.get("data", {}).get("ad_list")
                        or []
                    )
                    if materials:
                        print(f"[scraper] Capturado: {len(materials)} ads")
                        captured_materials.extend(materials)
                except Exception as e:
                    print(f"[scraper] Aviso ao parsear resposta: {e}")

        page.on("response", handle_response)

        print("[scraper] Navegando para TikTok Creative Center...")
        try:
            await page.goto(TIKTOK_CC_URL, wait_until="networkidle", timeout=60000)
        except Exception as e:
            print(f"[scraper] Timeout/erro na navegacao (pode ser ok): {e}")

        await page.wait_for_timeout(5000)

        if not captured_materials:
            print("[scraper] Tentando scroll e interacao para forcar carregamento...")
            await page.evaluate("window.scrollTo(0, 300)")
            await page.wait_for_timeout(4000)
            await page.evaluate("window.scrollTo(0, 800)")
            await page.wait_for_timeout(3000)
            # Tentar URL alternativa sem /inspiration/
            if not captured_materials:
                alt_url = (
                    "https://ads.tiktok.com/business/creativecenter/topads/pc/en"
                    "?period=7&region=BR&secondIndustry=14104000000"
                )
                print(f"[scraper] Tentando URL alternativa: {alt_url}")
                try:
                    await page.goto(alt_url, wait_until="networkidle", timeout=60000)
                except Exception:
                    pass
                await page.wait_for_timeout(8000)

        await browser.close()

    print(f"[scraper] Total capturado: {len(captured_materials)} ads")
    return captured_materials[:MAX_ADS]


# ─────────────────────────────────────────────
# MAIN SCRAPER FUNCTION
# ─────────────────────────────────────────────
async def run_scraper() -> dict:
    today = date.today()
    week_of = today - timedelta(days=today.weekday())

    print(f"[scraper] Iniciando para semana {week_of}")
    run_id = await create_scraper_run(week_of)

    try:
        materials = await scrape_tiktok_creative_center()

        if not materials:
            msg = "Nenhum anuncio capturado. Pagina pode ter mudado ou houve timeout."
            print(f"[scraper] AVISO: {msg}")
            await finish_scraper_run(run_id, "warning", 0, msg)
            return {"status": "warning", "message": msg, "items": 0}

        await deactivate_old_ads(week_of)

        saved = 0
        for rank, item in enumerate(materials, start=1):
            await upsert_trending_video(item, week_of, rank)
            saved += 1

        await finish_scraper_run(run_id, "success", saved)
        print(f"[scraper] Concluido: {saved} ads salvos para semana {week_of}")
        return {"status": "success", "items": saved, "week_of": str(week_of)}

    except Exception as e:
        error_msg = str(e)
        print(f"[scraper] ERRO: {error_msg}")
        await finish_scraper_run(run_id, "error", 0, error_msg)
        raise
