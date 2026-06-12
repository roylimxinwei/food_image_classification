// stores/detectionStore.ts
import { create } from 'zustand';

interface Classification {
  label: string;
  confidence: number;
}

interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface NutritionData {
  name: string;
  default_serving_size?: string;
  nutrition_per_100g: Record<string, string | number>;
  nutrition_per_serving?: Record<string, number>;
}

export interface ClassifiedDetection {
  item_id: number;
  confidence: number;
  bounding_box: BoundingBox;
  classification?: Classification;
  croppedImageBase64?: string;
  isManuallyEdited?: boolean;
  nutrition?: NutritionData;
}

interface DetectionStore {
  detections: ClassifiedDetection[];
  annotatedImage: string | null;
  
  setDetections: (detections: ClassifiedDetection[]) => void;
  setAnnotatedImage: (image: string | null) => void;
  updateDetectionName: (itemId: number, newName: string) => void;
  updateDetectionNutrition: (itemId: number, nutrition: NutritionData) => void;
  deleteDetection: (itemId: number) => void;
  clearAll: () => void;
}

export const useDetectionStore = create<DetectionStore>((set) => ({
  detections: [],
  annotatedImage: null,
  
  setDetections: (detections) => set({ detections }),
  
  setAnnotatedImage: (image) => set({ annotatedImage: image }),
  
  updateDetectionName: (itemId, newName) =>
    set((state) => ({
      detections: state.detections.map((d) =>
        d.item_id === itemId
          ? {
              ...d,
              classification: {
                label: newName.toLowerCase().replace(/\s+/g, '_'),
                confidence: d.classification?.confidence || 0,
              },
              isManuallyEdited: true,
              nutrition: undefined, // Clear nutrition when name changes
            }
          : d
      ),
    })),
  
  updateDetectionNutrition: (itemId, nutrition) =>
    set((state) => ({
      detections: state.detections.map((d) =>
        d.item_id === itemId ? { ...d, nutrition } : d
      ),
    })),
  
  deleteDetection: (itemId) =>
    set((state) => ({
      detections: state.detections.filter((d) => d.item_id !== itemId),
    })),
  
  clearAll: () => set({ detections: [], annotatedImage: null }),
}));