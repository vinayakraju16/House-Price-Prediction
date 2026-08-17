# Reviewed application screenshots

These captures were generated on August 16, 2026 from the healthy Docker Compose stack using
Chromium and active model version 2.2.0. The representative valuation uses 3 bedrooms, 2.5
bathrooms, 2,590 square feet, a 6,000-square-foot lot, and ZIP 98144. It contains no address,
API key, request log, or personal data.

## Desktop

- [Landing page](landing-desktop.png)
- [Empty valuation form](valuation-form-desktop.png)
- [Completed valuation, explanation, and comparables](valuation-result-desktop.png)
- [Model information](model-information-desktop.png)

## Mobile

- [Responsive landing page](landing-mobile.png)
- [Responsive completed valuation](valuation-result-mobile.png)

## Reproduce

```powershell
docker compose up --detach --wait --wait-timeout 180
cd frontend
npx playwright install chromium
npm run screenshots
cd ..
docker compose down
```

`PLAYWRIGHT_BASE_URL` can point the capture script at another reviewed environment. The script
fails if a page emits a browser runtime error or if the real prediction request fails.
