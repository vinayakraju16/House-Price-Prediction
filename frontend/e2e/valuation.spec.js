import { expect, test } from '@playwright/test';

test('serves the landing page and supports direct client routes', async ({ page }) => {
  await page.goto('/');

  await expect(
    page.getByRole('heading', { name: 'Know what your home could be worth.' }),
  ).toBeVisible();

  await page.getByRole('link', { name: /Get an estimate/i }).click();
  await expect(page).toHaveURL(/\/house-price$/);
  await expect(page.getByRole('heading', { name: 'Start with the details you know.' })).toBeVisible();

  await page.goto('/house-price');
  await expect(page.getByRole('button', { name: /Calculate estimate/i })).toBeVisible();
});

test('calculates a valuation in the browser and renders the enriched model response', async ({ page }) => {
  // The model runs client-side (src/model), so the page must not call a prediction API.
  const apiRequests = [];
  page.on('request', (request) => {
    if (/\/(api\/)?predict\/?$/.test(new URL(request.url()).pathname)) apiRequests.push(request.url());
  });
  await page.goto('/house-price');

  await page.getByLabel('Bedrooms').fill('3');
  await page.getByLabel('Bathrooms').fill('2.5');
  await page.getByLabel('Home size').fill('2590');
  await page.getByLabel('Lot size').fill('6000');
  await page.getByLabel('ZIP code').fill('98144');

  await page.getByRole('button', { name: /Calculate estimate/i }).click();

  await expect(page.getByText('Estimated value', { exact: true })).toBeVisible({ timeout: 15000 });
  await expect(page.locator('.result-card .price')).toHaveText(/^\$[\d,]+$/);
  await expect(page.getByText('90% empirical range', { exact: true })).toBeVisible();

  const factorSection = page.locator('section').filter({
    has: page.getByRole('heading', { name: 'What shaped this estimate' }),
  });
  // The UI shows the top 8 permutation-Shapley factors (Houseprice.jsx caps at .slice(0, 8));
  // the active King County model has more features than the legacy 5-field Seattle model did.
  await expect(factorSection.locator('.recent-item')).toHaveCount(8);

  const comparableSection = page.locator('section').filter({
    has: page.getByRole('heading', { name: 'Similar historical sales' }),
  });
  await expect(comparableSection.locator('.comparable-item')).toHaveCount(5);
  expect(apiRequests).toEqual([]);
});
