/**
 * Browser runtime for per-market models (export format 2), e.g. Wake County, NC.
 *
 * The JSON comes from `python -m ml.training.export_market_model`, which reads the fitted
 * scikit-learn pipeline: PropertyMarketFeatureEngineer -> median imputation + scaling and
 * one-hot encoding -> HistGradientBoosting trees, optionally wrapped in
 * MarketRelativeRegressor (price = exp(raw + log ZIP home value index)). Each step below
 * mirrors the Python arithmetic; marketPredictor.test.js checks parity against fixtures
 * scored by Python.
 */

export const MARKET_FEATURE_LABELS = {
  baths: 'Bathrooms',
  size: 'Living area',
  lot_size: 'Lot size',
  zip_code: 'ZIP code',
  stories: 'Stories',
  grade: 'Construction grade',
  condition: 'Condition',
  year_built: 'Year built',
  year_renovated: 'Year renovated',
  has_basement: 'Basement',
  has_central_air: 'Central air',
  heat_system: 'Heating',
  exterior_wall: 'Exterior wall',
  property_type: 'Home style',
  sale_year: 'Valuation year',
  sale_month: 'Valuation month',
  zhvi: 'Local home values (Zillow index)',
  zhvi_yoy: 'Local 12-month price trend',
  median_household_income: 'Area household income',
  owner_occupied_share: 'Area owner-occupancy',
};

const NUMERIC_RANGES = {
  baths: [0.5, 20], size: [1, 30_000], lot_size: [0, 5_000_000], stories: [1, 10],
  grade: [1, 7], condition: [0, 150], has_basement: [0, 1], has_central_air: [0, 1],
};
const INTEGER_FEATURES = new Set(['year_built', 'year_renovated', 'has_basement', 'has_central_air']);

export class MarketInputError extends Error {
  constructor(message, details) {
    super(message);
    this.name = 'PredictionInputError';
    this.details = details ?? { request: message };
  }
}

const isBlank = (value) => value === undefined || value === null || value === '';

function toNumber(value) {
  if (typeof value === 'number') return value;
  if (typeof value === 'boolean') return value ? 1 : 0;
  if (typeof value !== 'string') return null;
  const text = value.trim();
  return /^[+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?$/i.test(text) ? Number(text) : null;
}

export function normalizeZip(value) {
  const numeric = toNumber(value);
  if (numeric === null || !Number.isInteger(numeric) || numeric < 0 || numeric > 99999) return null;
  return String(numeric).padStart(5, '0');
}

/** Validate a form payload for this market; returns normalized values or throws. */
export function validateMarketPayload(model, payload, now = new Date()) {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new MarketInputError('The request body must be a JSON object.');
  }
  const currentYear = now.getUTCFullYear();
  const { numeric, categorical, required } = model.inputs;
  const details = {};
  const values = {};
  for (const name of required) if (isBlank(payload[name])) details[name] = 'This field is required.';

  if (!isBlank(payload.zip_code)) {
    const zip = normalizeZip(payload.zip_code);
    if (zip === null) details.zip_code = 'Must be a valid 5-digit ZIP code.';
    else if (!model.zip_codes.includes(zip)) details.zip_code = `Must be a ${model.market_name} ZIP code.`;
    else values.zip_code = zip;
  }
  const userNumeric = numeric.filter((name) => !model.inputs.excluded.includes(name));
  for (const name of userNumeric) {
    const raw = payload[name];
    if (isBlank(raw) || details[name]) continue;
    const number = toNumber(raw);
    if (number === null || !Number.isFinite(number)) details[name] = 'Must be a number.';
    else if (number < 0) details[name] = 'Cannot be negative.';
    else if (INTEGER_FEATURES.has(name) && !Number.isInteger(number)) details[name] = 'Must be a whole number.';
    else values[name] = number;
  }
  for (const [name, [low, high]] of Object.entries(NUMERIC_RANGES)) {
    if (name in values && !(low <= values[name] && values[name] <= high)) {
      details[name] = `Must be between ${low} and ${high.toLocaleString('en-US')}.`;
    }
  }
  if (values.size === 0) details.size = 'Must be greater than zero.';
  for (const name of ['year_built', 'year_renovated']) {
    if (name in values && values[name] !== 0 && !(values[name] >= 1800 && values[name] <= currentYear)) {
      details[name] = `Must be between 1800 and ${currentYear}.`;
    }
  }
  if (values.year_built === 0) details.year_built = `Must be between 1800 and ${currentYear}.`;
  if (values.year_renovated === 0) delete values.year_renovated;
  if ('year_renovated' in values && 'year_built' in values && values.year_renovated < values.year_built) {
    details.year_renovated = 'Cannot be earlier than year built.';
  }
  for (const name of categorical) {
    if (name === 'zip_code' || isBlank(payload[name])) continue;
    values[name] = String(payload[name]).trim();
  }
  const problems = Object.entries(details);
  if (problems.length) {
    throw new MarketInputError(problems.map(([field, problem]) => `${field}: ${problem}`).join('; '), details);
  }
  return values;
}

/** Training defaults, then validated values, then today's date and the ZIP's market data. */
export function buildMarketRow(model, payload, now = new Date()) {
  const values = validateMarketPayload(model, payload, now);
  const row = {};
  for (const name of model.features) row[name] = model.feature_defaults[name] ?? null;
  Object.assign(row, values);
  row.sale_year = now.getUTCFullYear();
  row.sale_month = now.getUTCMonth() + 1;
  const market = model.market_snapshot.zips[row.zip_code] ?? {};
  for (const name of model.inputs.market) row[name] = market[name] ?? null;
  return row;
}

const asNumber = (value) => (value === null || value === undefined ? Number.NaN : Number(value));

/** Port of PropertyMarketFeatureEngineer.transform for one row. */
export function engineerMarketFeatures(model, row) {
  const { numeric, categorical, market } = model.inputs;
  const out = {};
  for (const name of [...numeric, ...market]) out[name] = asNumber(row[name]);
  for (const name of categorical) out[name] = row[name] === null || row[name] === undefined ? null : String(row[name]);
  const saleYear = out.sale_year ?? Number.NaN;
  const built = out.year_built ?? Number.NaN;
  const renovated = out.year_renovated ?? Number.NaN;
  const size = out.size ?? Number.NaN;
  const lot = out.lot_size ?? Number.NaN;
  const age = Number.isNaN(saleYear - built) ? Number.NaN : Math.max(saleYear - built, 0);
  out.property_age = age;
  out.is_new_construction = Number.isFinite(age) ? (age <= 1 ? 1 : 0) : Number.NaN;
  out.was_renovated = Number.isFinite(renovated) ? 1 : 0;
  out.years_since_renovation = Number.isFinite(renovated) ? Math.max(saleYear - renovated, 0) : Number.NaN;
  out.living_lot_ratio = lot > 0 ? Math.min(Math.max(size / lot, 0), 20) : Number.NaN;
  return out;
}

export function transformMarketRow(model, row) {
  const engineered = engineerMarketFeatures(model, row);
  const { numeric, categorical } = model.preprocessing;
  const vector = [];
  numeric.columns.forEach((column, index) => {
    let value = engineered[column];
    if (Number.isNaN(value)) value = numeric.impute[index];
    vector.push((value - numeric.mean[index]) / numeric.scale[index]);
  });
  categorical.columns.forEach((column, index) => {
    const value = engineered[column] ?? categorical.impute[index];
    for (const category of categorical.categories[index]) vector.push(value === category ? 1 : 0);
  });
  return vector;
}

function rawPrediction(regressor, vector) {
  let raw = regressor.baseline;
  for (const tree of regressor.trees) {
    const { feature, threshold, left, right, value, missing_left: missingLeft } = tree;
    let node = 0;
    while (feature[node] !== -1) {
      const x = vector[feature[node]];
      const goLeft = Number.isNaN(x) ? missingLeft[node] === 1 : x <= threshold[node];
      node = goLeft ? left[node] : right[node];
    }
    raw += value[node];
  }
  return raw;
}

function logIndex(model, row) {
  const { relative } = model;
  const index = asNumber(row[relative.index_column]);
  if (Number.isFinite(index) && index > 0) return Math.log(index);
  const zip = row.zip_code === null || row.zip_code === undefined ? '' : String(row.zip_code).padStart(5, '0');
  return relative.zip_log_index[zip] ?? relative.global_log_index;
}

/** Dollar prediction for an already-built row. */
export function predictMarketRow(model, row) {
  const raw = rawPrediction(model.regressor, transformMarketRow(model, row));
  if (model.target_transform === 'zhvi_relative') return Math.exp(raw + logIndex(model, row));
  if (model.target_transform === 'log1p') return Math.expm1(raw);
  throw new Error(`Unsupported target transform: ${model.target_transform}`);
}

export function predictMarketPayload(model, payload, now = new Date()) {
  return predictMarketRow(model, buildMarketRow(model, payload, now));
}

/** Deterministic permutation Shapley values, identical to the exporter's Python version. */
export function explainMarketPrediction(model, row) {
  const reference = model.explanation_reference;
  const names = model.features;
  const { permutations } = model.shapley;
  const totals = new Array(names.length).fill(0);
  let baseSum = 0;
  let finalSum = 0;
  for (const order of permutations) {
    const pathRow = {};
    for (const name of names) pathRow[name] = name in reference ? reference[name] : row[name];
    let previous = predictMarketRow(model, pathRow);
    baseSum += previous;
    for (const index of order) {
      pathRow[names[index]] = row[names[index]];
      const next = predictMarketRow(model, pathRow);
      totals[index] += next - previous;
      previous = next;
    }
    finalSum += previous;
  }
  const factors = names.map((feature, index) => {
    const contribution = totals[index] / permutations.length;
    return {
      feature,
      label: MARKET_FEATURE_LABELS[feature] ?? feature.replaceAll('_', ' '),
      value: row[feature],
      reference_value: reference[feature] ?? null,
      contribution,
      direction: contribution >= 0 ? 'increased' : 'decreased',
    };
  });
  factors.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return {
    method: `permutation Shapley values (${permutations.length} deterministic paths)`,
    base_value: baseSum / permutations.length,
    prediction: finalSum / permutations.length,
    reference,
    top_factors: factors,
  };
}

/** Nearest recent qualified sales by baths, size, lot size and ZIP. */
export function findMarketComparables(model, comparables, row, limit = 5) {
  const { columns, log_scales: logScales, rows } = comparables;
  const distance = new Float64Array(rows);
  const bounds = model.typical_feature_ranges.baths;
  const bathScale = Math.max(bounds.p95 - bounds.p05, 1);
  const baths = Number(row.baths);
  for (let i = 0; i < rows; i += 1) distance[i] += Math.abs(columns.baths[i] - baths) / bathScale;
  for (const [feature, weight] of [['size', 2], ['lot_size', 0.5]]) {
    const scale = Math.max(logScales[feature], 1);
    const query = Math.log1p(Number(row[feature]));
    const data = columns[feature];
    for (let i = 0; i < rows; i += 1) distance[i] += (weight * Math.abs(Math.log1p(data[i]) - query)) / scale;
  }
  for (let i = 0; i < rows; i += 1) if (columns.zip_code[i] !== row.zip_code) distance[i] += 2;
  const nearest = [];
  for (let i = 0; i < rows; i += 1) {
    if (nearest.length < limit || distance[i] < distance[nearest[nearest.length - 1]]) {
      nearest.push(i);
      nearest.sort((a, b) => distance[a] - distance[b] || a - b);
      if (nearest.length > limit) nearest.pop();
    }
  }
  return nearest.map((i) => ({
    price: columns.price[i],
    baths: columns.baths[i],
    size: columns.size[i],
    lot_size: columns.lot_size[i],
    zip_code: columns.zip_code[i],
    property_type: columns.property_type[i],
    sale_date: columns.sale_date[i],
    year_built: columns.year_built[i],
    similarity: 1 / (1 + distance[i]),
    source: model.comparable_source,
  }));
}

function inputConfidence(model, row) {
  const ranges = model.typical_feature_ranges;
  const unusual = ['baths', 'size', 'lot_size'].filter((name) => {
    const bounds = ranges[name];
    const value = Number(row[name]);
    return bounds && !(bounds.p05 <= value && value <= bounds.p95);
  });
  if (unusual.length === 0) return { level: 'High', message: 'All entered values are within typical ranges seen during training.', unusual_features: unusual };
  if (unusual.length <= 2) return { level: 'Medium', message: 'Some entered values are uncommon in the training data.', unusual_features: unusual };
  return { level: 'Low', message: 'Several entered values are outside typical training ranges.', unusual_features: unusual };
}

/** The ZIP's latest market conditions, shown next to the estimate. */
export function marketContext(model, zip) {
  const snapshot = model.market_snapshot.zips[zip];
  if (!snapshot) return null;
  return { month: model.market_snapshot.month, zip, ...snapshot };
}

const round2 = (value) => Math.round(value * 100) / 100;

/** Same response shape as the King County `estimate`, plus market metadata and context. */
export function estimateMarket(model, comparables, payload, now = new Date()) {
  const row = buildMarketRow(model, payload, now);
  const prediction = predictMarketRow(model, row);
  const interval = model.prediction_interval;
  const low = prediction * interval.lower_ratio;
  const high = prediction * interval.upper_ratio;
  const explanation = explainMarketPrediction(model, row);
  return {
    prediction: round2(prediction),
    range: {
      low: round2(low),
      high: round2(high),
      margin: round2(Math.max(prediction - low, high - prediction)),
      method: interval.method,
      nominal_coverage: round2(interval.nominal_coverage),
    },
    confidence: inputConfidence(model, row),
    model_version: model.model_version,
    market: model.market,
    market_name: model.market_name,
    market_context: marketContext(model, row.zip_code),
    model_factors: explanation.top_factors,
    top_factors: explanation.top_factors,
    explanation,
    comparables: findMarketComparables(model, comparables, row),
    disclaimer: model.disclaimer,
  };
}
