import { chromium } from '@playwright/test';

const baseURL = process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:3000';
const outputDirectory = '../docs/screenshots';
const property = {
  Bedrooms: '3',
  Bathrooms: '2.5',
  'Home size': '2590',
  'Lot size': '6000',
  'ZIP code': '98144',
};

async function openPage(context, path) {
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto(`${baseURL}${path}`, { waitUntil: 'networkidle' });
  return { page, pageErrors };
}

async function submitValuation(page) {
  for (const [label, value] of Object.entries(property)) {
    await page.getByLabel(label).fill(value);
  }
  const responsePromise = page.waitForResponse(
    (response) => response.url().endsWith('/api/predict/') && response.request().method() === 'POST',
  );
  await page.getByRole('button', { name: /Calculate estimate/i }).click();
  const response = await responsePromise;
  if (!response.ok()) {
    throw new Error(`Prediction request failed with HTTP ${response.status()}`);
  }
  await page.getByRole('heading', { name: 'Similar historical records' }).waitFor();
}

async function capture(page, filename) {
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(100);
  await page.screenshot({
    path: `${outputDirectory}/${filename}`,
    fullPage: true,
    animations: 'disabled',
    caret: 'hide',
  });
}

function assertNoPageErrors(name, errors) {
  if (errors.length) {
    throw new Error(`${name} emitted browser errors: ${errors.join('; ')}`);
  }
}

const browser = await chromium.launch();

try {
  const desktop = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    deviceScaleFactor: 1,
  });

  const landing = await openPage(desktop, '/');
  await capture(landing.page, 'landing-desktop.png');
  assertNoPageErrors('Desktop landing page', landing.pageErrors);
  await landing.page.close();

  const valuation = await openPage(desktop, '/house-price');
  await capture(valuation.page, 'valuation-form-desktop.png');
  await submitValuation(valuation.page);
  await capture(valuation.page, 'valuation-result-desktop.png');
  assertNoPageErrors('Desktop valuation page', valuation.pageErrors);
  await valuation.page.close();

  const modelInformation = await openPage(desktop, '/about');
  await modelInformation.page.getByLabel('Model performance').waitFor();
  await capture(modelInformation.page, 'model-information-desktop.png');
  assertNoPageErrors('Model information page', modelInformation.pageErrors);
  await modelInformation.page.close();
  await desktop.close();

  const mobile = await browser.newContext({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 1,
    isMobile: true,
    hasTouch: true,
  });

  const mobileLanding = await openPage(mobile, '/');
  await capture(mobileLanding.page, 'landing-mobile.png');
  assertNoPageErrors('Mobile landing page', mobileLanding.pageErrors);
  await mobileLanding.page.close();

  const mobileValuation = await openPage(mobile, '/house-price');
  await submitValuation(mobileValuation.page);
  await capture(mobileValuation.page, 'valuation-result-mobile.png');
  assertNoPageErrors('Mobile valuation page', mobileValuation.pageErrors);
  await mobileValuation.page.close();
  await mobile.close();
} finally {
  await browser.close();
}

console.log(`Captured reviewed application screenshots in ${outputDirectory}`);
