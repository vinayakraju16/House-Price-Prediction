import { estimate } from './predictor';

export { PredictionInputError } from './predictor';

/**
 * The exported model and comparable sales (~2 MB of JSON) are loaded with dynamic imports,
 * so Vite emits them as separate files that the browser downloads only when someone asks
 * for an estimate or opens the model page, and then caches.
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

export const loadModel = cached(() => import('./king-county-3.0.0.json'));
export const loadComparables = cached(() => import('./comparables-3.0.0.json'));

/** Drop-in replacement for the old POST /predict/ request. */
export async function requestEstimate(payload, now = new Date()) {
  const [model, comparables] = await Promise.all([loadModel(), loadComparables()]);
  return estimate(model, comparables, payload, now);
}

/** Drop-in replacement for the old GET /model-info/ request. */
export async function requestModelInfo() {
  return (await loadModel()).model_info;
}
