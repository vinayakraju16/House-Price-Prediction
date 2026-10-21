import fixtures from './__fixtures__/parity-wake-county-1.0.0.json';
import comparables from './wake-county-comparables-1.0.0.json';
import model from './wake-county-1.0.0.json';
import {
  buildMarketRow,
  estimateMarket,
  explainMarketPrediction,
  findMarketComparables,
  predictMarketRow,
  validateMarketPayload,
} from './marketPredictor';

// Fixtures come from `python -m ml.training.export_market_model --fixtures`: random
// properties scored by the real scikit-learn pipeline and the exporter's Python port of
// the explanation and comparable search.
const fixtureDate = new Date(Date.UTC(fixtures.sale_year, fixtures.sale_month - 1, 15));
const sampleHome = { baths: '2.5', size: '2200', lot_size: '10890', zip_code: '27615' };

function validationError(payload) {
  try {
    validateMarketPayload(model, payload, fixtureDate);
  } catch (error) {
    return error;
  }
  throw new Error('expected validation to fail');
}

describe('parity with the scikit-learn pipeline', () => {
  test.each(fixtures.cases.map((item, index) => [index, item]))('case %i prediction', (_, item) => {
    const row = buildMarketRow(model, item.payload, fixtureDate);
    expect(Math.abs(predictMarketRow(model, row) - item.prediction) / item.prediction).toBeLessThan(1e-9);
  });

  test('explanation factors match Python and add up to the prediction', () => {
    for (const item of fixtures.cases.slice(0, 10)) {
      const row = buildMarketRow(model, item.payload, fixtureDate);
      const explanation = explainMarketPrediction(model, row);
      const total = explanation.top_factors.reduce((sum, factor) => sum + factor.contribution, 0);
      expect(explanation.base_value).toBeCloseTo(item.base_value, 4);
      expect(explanation.base_value + total).toBeCloseTo(item.prediction, 4);
      for (const factor of explanation.top_factors) {
        expect(factor.contribution).toBeCloseTo(item.contributions[factor.feature], 4);
      }
    }
  }, 60000);

  test('comparable sales match Python', () => {
    for (const item of fixtures.cases) {
      const found = findMarketComparables(model, comparables, buildMarketRow(model, item.payload, fixtureDate));
      found.forEach((property, index) => {
        expect(property.similarity).toBeCloseTo(item.comparable_similarity[index], 10);
      });
      // numpy's argsort may order exactly tied records differently, including which of two
      // records tied for the last place is kept, so compare the records ranked above the tie.
      const cutoff = item.comparable_similarity[item.comparable_similarity.length - 1];
      const aboveCutoff = (prices, similarity) => prices.filter((_, i) => similarity[i] > cutoff + 1e-12).sort();
      expect(aboveCutoff(found.map((property) => property.price), found.map((property) => property.similarity)))
        .toEqual(aboveCutoff(item.comparable_prices, item.comparable_similarity));
    }
  });
});

describe('estimateMarket', () => {
  test('returns the estimate shape the page expects, with a percentage range', () => {
    const result = estimateMarket(model, comparables, sampleHome, fixtureDate);
    expect(result.prediction).toBeGreaterThan(100_000);
    expect(result.range.low).toBeCloseTo(result.prediction * model.prediction_interval.lower_ratio, 0);
    expect(result.range.high).toBeCloseTo(result.prediction * model.prediction_interval.upper_ratio, 0);
    expect(result.market).toBe('wake-county');
    expect(result.market_context.zip).toBe('27615');
    expect(result.comparables).toHaveLength(5);
    expect(result.top_factors).toHaveLength(model.features.length);
  });

  test('estimates follow the ZIP home value index', () => {
    const row = buildMarketRow(model, sampleHome, fixtureDate);
    const base = predictMarketRow(model, row);
    expect(predictMarketRow(model, { ...row, zhvi: row.zhvi * 1.05 }) / base).toBeGreaterThan(1.0);
  });

  test('uses market data for the ZIP that was entered', () => {
    const row = buildMarketRow(model, sampleHome, fixtureDate);
    expect(row.zhvi).toBe(model.market_snapshot.zips['27615'].zhvi);
    expect(row.sale_year).toBe(fixtures.sale_year);
  });
});

describe('validation', () => {
  test('requires the basics and a covered ZIP', () => {
    expect(validationError({}).details).toMatchObject({
      baths: 'This field is required.', size: 'This field is required.',
      lot_size: 'This field is required.', zip_code: 'This field is required.',
    });
    expect(validationError({ ...sampleHome, zip_code: '98144' }).details.zip_code).toMatch(/Wake County/);
  });

  test('rejects impossible values and treats renovation year 0 as never', () => {
    expect(validationError({ ...sampleHome, size: '-5' }).details.size).toBe('Cannot be negative.');
    expect(validationError({ ...sampleHome, year_built: '2000', year_renovated: '1990' }).details.year_renovated)
      .toBe('Cannot be earlier than year built.');
    expect(validateMarketPayload(model, { ...sampleHome, year_renovated: '0' }, fixtureDate).year_renovated).toBeUndefined();
  });
});
