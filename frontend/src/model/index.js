import markets from './markets.json';
import { estimateMarket, predictMarketPayload } from './marketPredictor';
import { estimate, predictPayload, PredictionInputError } from './predictor';

export { PredictionInputError } from './predictor';

/**
 * Each market's exported model and comparable sales (~1-2 MB of JSON) are loaded with
 * dynamic imports, so Vite emits them as separate files that the browser downloads only
 * for the market of the ZIP code being estimated, and then caches. `markets.json` (1 KB)
 * is the routing table written by `python -m ml.training.export_market_model`.
 */
function cached(loader) {
  let pending = null;
  return () => {
    if (!pending) {
      pending = loader().then((module) => module.default).catch((error) => {
        pending = null;
        throw error;
      });
    }
    return pending;
  };
}

const LOADERS = {
  'king-county': {
    model: cached(() => import('./king-county-3.1.0.json')),
    comparables: cached(() => import('./comparables-3.1.0.json')),
    estimate,
    predict: predictPayload,
  },
  'wake-county': {
    model: cached(() => import('./wake-county-1.0.0.json')),
    comparables: cached(() => import('./wake-county-comparables-1.0.0.json')),
    estimate: estimateMarket,
    predict: predictMarketPayload,
  },
};

export const MARKETS = markets.markets;
export const DEFAULT_MARKET = 'king-county';

/** The market whose model covers this ZIP code, or null. */
export function marketForZip(zip) {
  const text = String(zip ?? '').trim();
  if (!/^\d{5}$/.test(text)) return null;
  const number = Number(text);
  return MARKETS.find((market) => (market.zip_codes ?? []).includes(text)
    || (market.zip_ranges ?? []).some(([low, high]) => number >= low && number <= high)) ?? null;
}

/** Every trained ZIP with its market, place name and training coverage. */
export const ZIP_DIRECTORY = MARKETS.flatMap((market) => (market.zips ?? []).map((entry) => ({
  ...entry, market: market.id, marketName: market.name, marketShortName: market.short_name,
})));
export const MIN_ZIP_SALES = markets.min_zip_sales ?? 30;

/** Directory entry for a ZIP code, or null when the model has no sales there. */
export function zipInfo(zip) {
  const text = String(zip ?? '').trim();
  return ZIP_DIRECTORY.find((entry) => entry.zip === text) ?? null;
}

/** The well-covered ZIPs of a market with the most training sales. */
export function popularZips(marketId, count = 4) {
  return ZIP_DIRECTORY
    .filter((entry) => entry.market === marketId && entry.well_covered)
    .sort((a, b) => b.sales - a.sales)
    .slice(0, count);
}

export function coverageMessage() {
  return `We currently cover ${MARKETS.map((market) => market.name).join(' and ')}.`;
}

export const loadModel = (marketId = DEFAULT_MARKET) => LOADERS[marketId].model();
export const loadComparables = (marketId = DEFAULT_MARKET) => LOADERS[marketId].comparables();

/** Price a payload with an already-loaded model of the given market (what-if sliders). */
export function predictWithModel(marketId, model, payload, now = new Date()) {
  return LOADERS[marketId].predict(model, payload, now);
}

/** Route the payload to the model for its ZIP code and return the full estimate. */
export async function requestEstimate(payload, now = new Date()) {
  const zip = String(payload?.zip_code ?? '').trim();
  const market = zip === '' ? MARKETS.find((item) => item.id === DEFAULT_MARKET) : marketForZip(zip);
  if (!market) {
    const message = `zip_code: ${coverageMessage()}`;
    throw new PredictionInputError(message, { zip_code: coverageMessage() });
  }
  const loader = LOADERS[market.id];
  const [model, comparables] = await Promise.all([loader.model(), loader.comparables()]);
  return { market: market.id, market_name: market.name, ...loader.estimate(model, comparables, payload, now) };
}

/** Model summaries for every market (About page). */
export async function requestModelInfo(marketId = DEFAULT_MARKET) {
  return (await loadModel(marketId)).model_info;
}
