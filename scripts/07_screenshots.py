"""Step 7: screenshots of the running app for the video and README (start the app first).

    python -m graphrag.app        (in another terminal; GRAPHRAG_NO_BROWSER=1 to stay headless)
    python scripts/07_screenshots.py
"""
import os
import time

from playwright.sync_api import sync_playwright

import _path  # noqa: F401
from graphrag import config

URL = f"http://127.0.0.1:{os.environ.get('GRAPHRAG_PORT', 7861)}"
Q_LOCAL = os.environ.get("Q_LOCAL", "How do dry conditions in South Australia connect to barley and wheat production?")


def shot(page, name, wait=4):
    time.sleep(wait)
    page.screenshot(path=os.path.join(config.RESULTS, name))
    print("wrote", name)


def wait_text(page, sel, text, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if text in page.locator(sel).first.inner_text():
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(URL)
        page.wait_for_selector("textarea", timeout=120000)
        # Ask tab: local GraphRAG on a connected question
        page.get_by_label("GraphRAG: local (entities + relationships)").check()
        page.get_by_label("Your question").fill(Q_LOCAL)
        page.get_by_role("button", name="Ask", exact=True).click()
        wait_text(page, "#answer", "Answered in")
        shot(page, "app_local.png", 6)
        # Auto mode on the same question -> hybrid (README hero image)
        page.get_by_label("Auto (let the app choose)").check()
        page.get_by_role("button", name="Ask", exact=True).click()
        time.sleep(2)
        wait_text(page, "#answer", "Answered in")
        shot(page, "app_ask.png", 6)
        # Compare tab
        page.get_by_role("tab", name="Compare").click()
        page.get_by_label("Question to compare").fill(Q_LOCAL)
        page.get_by_role("button", name="Compare vector RAG and GraphRAG").click()
        time.sleep(5)
        t0 = time.time()
        time.sleep(3)
        while time.time() - t0 < 600 and page.get_by_text("processing").count() > 0:   # both answers finished
            time.sleep(2)
        shot(page, "app_compare.png", 6)
        # Explore tab
        page.get_by_role("tab", name="Explore the graph").click()
        shot(page, "app_explore.png", 8)
        # Topics tab
        page.get_by_role("tab", name="Topics").click()
        shot(page, "app_topics.png", 8)
        # Cypher tab
        page.get_by_role("tab", name="Cypher").click()
        page.get_by_role("button", name="Run", exact=True).click()
        shot(page, "app_cypher.png", 4)
        b.close()


if __name__ == "__main__":
    main()
