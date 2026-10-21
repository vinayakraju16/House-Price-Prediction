import { fireEvent, render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import comparables from '../model/comparables-3.1.0.json';
import model from '../model/king-county-3.1.0.json';
import { estimateMarket, predictMarketPayload } from '../model/marketPredictor';
import { estimate, predictPayload } from '../model/predictor';
import wakeComparables from '../model/wake-county-comparables-1.0.0.json';
import wakeModel from '../model/wake-county-1.0.0.json';
import Houseprice from './Houseprice';

const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
const basics = { beds: '3', baths: '2.5', size: '2590', lot_size: '6000', zip_code: '98144' };

beforeEach(() => {
  localStorage.clear();
  // The estimate is computed in the browser, so no network request should be made.
  global.fetch = vi.fn();
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('calculates an estimate, local factors, and comparable records in the browser', async () => {
  render(<Houseprice />);

  fireEvent.change(screen.getByLabelText(/Bedrooms/i), { target: { value: basics.beds } });
  fireEvent.change(screen.getByLabelText(/Bathrooms/i), { target: { value: basics.baths } });
  fireEvent.change(screen.getByLabelText(/Home size/i), { target: { value: basics.size } });
  fireEvent.change(screen.getByLabelText(/^Lot size/i), { target: { value: basics.lot_size } });
  fireEvent.change(screen.getByLabelText(/^ZIP code/i), { target: { value: basics.zip_code } });
  fireEvent.click(screen.getByRole('button', { name: /Calculate estimate/i }));

  expect(await screen.findByText('90% empirical range', {}, { timeout: 15000 })).toBeInTheDocument();
  const expected = estimate(model, comparables, basics, new Date());
  // Shown in the result card and in Recent estimates (and in the what-if panel once it loads).
  expect(screen.getAllByText(money.format(expected.prediction)).length).toBeGreaterThanOrEqual(2);
  expect(screen.getByRole('heading', { name: /What shaped this estimate/i })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: /Similar historical sales/i })).toBeInTheDocument();
  expect(global.fetch).not.toHaveBeenCalled();
}, 20000);

test('shows the same validation message the API used to return', async () => {
  render(<Houseprice />);

  fireEvent.click(screen.getByRole('button', { name: /Use sample/i }));
  fireEvent.change(screen.getByLabelText(/Bedrooms/i), { target: { value: '25' } });
  fireEvent.click(screen.getByRole('button', { name: /Calculate estimate/i }));

  expect(await screen.findByRole('alert', {}, { timeout: 15000 })).toHaveTextContent('beds: Must be between 0 and 20.');
}, 20000);

test('what-if sliders re-price the home instantly', async () => {
  render(<Houseprice />);

  fireEvent.change(screen.getByLabelText(/Bedrooms/i), { target: { value: basics.beds } });
  fireEvent.change(screen.getByLabelText(/Bathrooms/i), { target: { value: basics.baths } });
  fireEvent.change(screen.getByLabelText(/Home size/i), { target: { value: basics.size } });
  fireEvent.change(screen.getByLabelText(/^Lot size/i), { target: { value: basics.lot_size } });
  fireEvent.change(screen.getByLabelText(/^ZIP code/i), { target: { value: basics.zip_code } });
  fireEvent.click(screen.getByRole('button', { name: /Calculate estimate/i }));

  expect(await screen.findByRole('heading', { name: /What if/i }, { timeout: 15000 })).toBeInTheDocument();
  expect(screen.getByText('Same as your estimate')).toBeInTheDocument();

  fireEvent.change(screen.getByRole('slider', { name: /Living area/i }), { target: { value: '3500' } });
  const expected = predictPayload(model, { ...basics, size: '3500' }, new Date());
  expect(screen.getByTestId('whatif-estimate')).toHaveTextContent(money.format(expected));
  expect(screen.queryByText('Same as your estimate')).not.toBeInTheDocument();
}, 20000);

test('routes a Wake County ZIP to the Wake model, without a bedrooms field', async () => {
  render(<Houseprice />);

  fireEvent.change(screen.getByLabelText(/^ZIP code/i), { target: { value: '27615' } });
  expect(screen.getByTestId('market-note')).toHaveTextContent('Using the Wake County, NC model');
  expect(screen.queryByLabelText(/Bedrooms/i)).not.toBeInTheDocument();

  const wake = { baths: '2.5', size: '2200', lot_size: '10890', zip_code: '27615' };
  fireEvent.change(screen.getByLabelText(/Bathrooms/i), { target: { value: wake.baths } });
  fireEvent.change(screen.getByLabelText(/Home size/i), { target: { value: wake.size } });
  fireEvent.change(screen.getByLabelText(/^Lot size/i), { target: { value: wake.lot_size } });
  fireEvent.click(screen.getByRole('button', { name: /Calculate estimate/i }));

  expect(await screen.findByRole('heading', { name: /Market in ZIP 27615/i }, { timeout: 15000 })).toBeInTheDocument();
  const payload = { ...wake, property_type: '', stories: '', grade: '', condition: '', exterior_wall: '', year_built: '', year_renovated: '', has_basement: '', has_central_air: '' };
  const expected = estimateMarket(wakeModel, wakeComparables, payload, new Date());
  expect(screen.getAllByText(money.format(expected.prediction)).length).toBeGreaterThanOrEqual(2);
  expect(screen.getByText(/Nearest Wake County qualified sales/i)).toBeInTheDocument();
  expect(global.fetch).not.toHaveBeenCalled();
}, 20000);

test('explains which markets are covered for other ZIP codes', async () => {
  render(<Houseprice />);

  fireEvent.change(screen.getByLabelText(/^ZIP code/i), { target: { value: '10001' } });
  expect(screen.getByRole('status')).toHaveTextContent('We currently cover Seattle (King County, WA) and Wake County, NC.');
});

test('Wake what-if sliders re-price with the Wake model, including the grade steps', async () => {
  render(<Houseprice />);

  fireEvent.change(screen.getByLabelText(/^ZIP code/i), { target: { value: '27615' } });
  fireEvent.click(screen.getByRole('button', { name: /Use sample/i }));
  fireEvent.click(screen.getByRole('button', { name: /Calculate estimate/i }));

  expect(await screen.findByRole('heading', { name: /What if/i }, { timeout: 15000 })).toBeInTheDocument();
  expect(screen.getByText('Same as your estimate')).toBeInTheDocument();

  // Grade steps are [2, 3, 3.1, 4, 5, 6]; position 3 is grade 4 (B, above average).
  fireEvent.change(screen.getByRole('slider', { name: /Construction grade/i }), { target: { value: '3' } });
  const sample = { zip_code: '27615', baths: '2.5', size: '2200', lot_size: '10890', property_type: 'Conventional', stories: '2', grade: 4, condition: '82', exterior_wall: 'Z', year_built: '2005', year_renovated: '', has_basement: 0, has_central_air: '1' };
  const expected = predictMarketPayload(wakeModel, sample, new Date());
  expect(screen.getByTestId('whatif-estimate')).toHaveTextContent(money.format(expected));
  expect(screen.queryByText('Same as your estimate')).not.toBeInTheDocument();
}, 20000);

test('popular ZIP picks fill the ZIP, switch the market and show training coverage', () => {
  render(<Houseprice />);

  fireEvent.click(screen.getByRole('button', { name: /Use ZIP 27587, Wake Forest/i }));
  expect(screen.getByLabelText(/^ZIP code/i)).toHaveValue('27587');
  expect(screen.getByTestId('market-note')).toHaveTextContent('Using the Wake County, NC model');
  expect(screen.getByTestId('zip-coverage')).toHaveTextContent(/Wake Forest, Wake County · [\d,]+ training sales · typical value \$/);
  expect(screen.queryByLabelText(/Bedrooms/i)).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: /Use ZIP 98103/i }));
  expect(screen.getByTestId('zip-coverage')).toHaveTextContent(/Fremont \/ Wallingford \/ Greenwood, Seattle/);
  expect(screen.getByLabelText(/Bedrooms/i)).toBeInTheDocument();
});

test('flags covered-market ZIPs with little or no training data', () => {
  render(<Houseprice />);

  fireEvent.change(screen.getByLabelText(/^ZIP code/i), { target: { value: '98113' } });
  expect(screen.getByTestId('zip-coverage')).toHaveTextContent(/^Only \d+ training sales in this ZIP; the estimate leans on nearby areas/);
  fireEvent.change(screen.getByLabelText(/^ZIP code/i), { target: { value: '98151' } });
  expect(screen.getByTestId('zip-coverage')).toHaveTextContent('No training sales in this ZIP');
});
