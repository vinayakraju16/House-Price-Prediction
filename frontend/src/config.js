const configuredApiUrl = import.meta.env.VITE_API_URL;

export const API_URL = (configuredApiUrl || 'https://api.vinayakraju.dev').replace(/\/$/, '');

//export const API_URL = (configuredApiUrl || 'http://localhost:8000').replace(/\/$/, '');


