import { API_V1 } from "@/config/api";

export interface PredictionResult {
    rank: number;
    label: string;
    confidence: number;
}

export interface ClassificationResponse {
    success: boolean;
    predictions: PredictionResult[];
    top_prediction: string;
    model_name: string;
}

export async function classifyFood(imageBase64: string): Promise<ClassificationResponse> {
    const response = await fetch(`${API_V1}/classify`, {
        method:'POST',
        headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({
            image_base64: imageBase64,
            topk: 5,
         }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({detail: 'Unknown error'}))
        throw new Error(error.detail || `HTTP ${response.status}`);
    }
    return response.json();
}

// Simple health check to verify connection
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_V1.replace('/api/v1', '')}/health`, {
      headers: {
        'ngrok-skip-browser-warning': 'true',
      },
    });
    return response.ok;
  } catch {
    return false;
  }
}