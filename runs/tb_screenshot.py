"""Capture TensorBoard Time Series scalar cards (one image per tag) with headless Chrome.

Usage: python tb_screenshot.py <out_prefix> <smoothing> <tag> [tag ...]
"""

import sys

from playwright.sync_api import sync_playwright

URL = "http://localhost:6006/?darkMode=false#timeseries"


def main(out_prefix: str, smoothing: str, tags: list[str]) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/local/bin/google-chrome")
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
        page.goto(URL)
        page.wait_for_selector("card-view", timeout=60_000)
        outliers = page.locator("mat-checkbox", has_text="Ignore outliers").locator("input")
        if outliers.is_checked():
            outliers.click()
        smoothing_input = page.locator("metrics-dashboard-settings input[type='number']").first
        smoothing_input.fill(smoothing)
        smoothing_input.press("Enter")
        page.locator("metrics-dashboard-right-pane button").first.click()
        for tag in tags:
            page.locator("metrics-tag-filter input").fill(f"^{tag}$")
            page.keyboard.press("Escape")
            page.wait_for_timeout(2500)
            card = page.locator("card-view:visible").first
            card.locator("button[aria-label*='full size' i]").first.click()
            page.wait_for_timeout(1500)
            fit = card.locator("button[aria-label='Fit line chart domains to data']")
            if fit.is_enabled():
                fit.click()
            page.wait_for_timeout(1500)
            card.screenshot(path=f"{out_prefix}_{tag}.png")
            card.locator("button[aria-label*='full size' i]").first.click()
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
