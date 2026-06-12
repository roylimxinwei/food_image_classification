import { API_V1 } from "@/config/api";

export interface NutritionSearchResult {
  index: number;
  name: string;
}

export interface NutritionSearchResponse {
  success: boolean;
  query: string;
  count: number;
  results: NutritionSearchResult[];
}

export interface NutritionDetails {
  success: boolean;
  name: string;
  description?: string;
  default_serving_size?: string;
  nutrition_per_100g: Record<string, string>;
  nutrition_per_serving?: Record<string, number>;
  extra_info?: Record<string, string>;
  source?: 'database' | 'web';
}

// services/nutritionApi.ts or utils/nutritionOrder.ts

export const NUTRIENT_ORDER = [
  "Energy (kcal)",
  "Protein (g)",
  "Total Fat (g)",
  "Saturated Fat (g)",
  "Trans Fat (g)",
  "Carbohydrate (g)",
  "Sugar (g)",
  "Added Sugar (g)",
  "Dietary Fibre (g)",
  "Sodium (mg)",
  "Potassium (mg)",
  "Calcium (mg)",
  "Iron (mg)",
];

export function sortNutrients(
  nutrition: Record<string, string | number>
): [string, string | number][] {
  const entries = Object.entries(nutrition);
  
  return entries.sort((a, b) => {
    const indexA = NUTRIENT_ORDER.indexOf(a[0]);
    const indexB = NUTRIENT_ORDER.indexOf(b[0]);
    
    // If not in the list, put at the end
    const orderA = indexA === -1 ? 999 : indexA;
    const orderB = indexB === -1 ? 999 : indexB;
    
    return orderA - orderB;
  });
}

export const ADDITIONAL_INFO_ORDER = [
    "Food Group",
    "Food Subgroup",
    "Edible Portion",
    "Default Serving Size",
    "Alternative Serving Size(s)",
    "Source of Data",

]

export function sortAdditionalInfo(
  extraInfo: Record<string, string>
): [string, string][] {
  const entries = Object.entries(extraInfo);
  
  return entries.sort((a, b) => {
    const indexA = ADDITIONAL_INFO_ORDER.indexOf(a[0]);
    const indexB = ADDITIONAL_INFO_ORDER.indexOf(b[0]);
    
    // If not in the list, put at the end
    const orderA = indexA === -1 ? 999 : indexA;
    const orderB = indexB === -1 ? 999 : indexB;
    
    return orderA - orderB;
  });
}

/**
 * Search for food items by name
 */
export async function searchNutrition(
  query: string,
  maxResults: number = 5
): Promise<NutritionSearchResponse> {
  const params = new URLSearchParams({
    query,
    max_results: maxResults.toString(),
  });

  const response = await fetch(`${API_V1}/nutrition/search?${params}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Search failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Get detailed nutrition info for a specific search result
 */
export async function getNutritionDetails(
  query: string,
  index: number
): Promise<NutritionDetails> {
  const params = new URLSearchParams({ query });

  const response = await fetch(`${API_V1}/nutrition/details/${index}?${params}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to get details' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Quick lookup - search and get details in one call
 * Uses the first search result automatically
 */
export async function quickNutritionLookup(
  query: string,
  resultIndex: number = 0
): Promise<NutritionDetails> {
  const params = new URLSearchParams({
    query,
    result_index: resultIndex.toString(),
  });

  const response = await fetch(`${API_V1}/nutrition/quick?${params}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Lookup failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Check if food name exist in database cache
 */
export async function getCachedNutrition(name: string): Promise<NutritionDetails | null> {
  try {
    const response = await fetch(
      `${API_V1}/nutrition/cached?name=${encodeURIComponent(name)}`
    );
    
    if (response.status === 404) {
      // Not in cache
      return null;
    }
    
    if (!response.ok) {
      throw new Error('Cache lookup failed');
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    // On any error, return null to fall back to web scraping
    console.log('Cache miss or error:', error);
    return null;
  }
}