import { API_V1 } from "@/config/api";

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface SegmentationDetection {
  item_id: number;
  confidence: number;
  bounding_box: BoundingBox;
}

export interface SegmentationResponse {
  success: boolean;
  prompt: string;
  num_detections: number;
  detections: SegmentationDetection[];
  annotated_image_base64?: string;
  message?: string;
}

export interface SegmentationStatusResponse {
  available: boolean;
  message: string;
}

// =========== Extending for classification ===========

export interface ClassifiedDetection extends SegmentationDetection {
    classification? : {
        label: string;
        confidence: number;
    };
    cropped_image_base64?: string;
}

export interface SegmentationWithClassification extends SegmentationResponse {
  classified_detections?: ClassifiedDetection[];
}

// ============ ADD SEGMENTATION FUNCTIONS ============

/**
 * Check if segmentation service is available
 */
export async function checkSegmentationStatus(): Promise<SegmentationStatusResponse> {
  const response = await fetch(`${API_V1}/segment/status`);
  
  if (!response.ok) {
    throw new Error('Failed to check segmentation status');
  }
  
  return response.json();
}

/**
 * Segment food items in an image
 */
export async function segmentFood(
  imageBase64: string,
  prompt: string = 'food',
  confidenceThreshold: number = 0.5
): Promise<SegmentationResponse> {
  const response = await fetch(`${API_V1}/segment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: imageBase64,
      prompt,
      confidence_threshold: confidenceThreshold,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Segmentation failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}