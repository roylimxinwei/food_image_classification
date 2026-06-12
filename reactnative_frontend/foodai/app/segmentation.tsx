// app/segmentation.tsx
import { 
  View, 
  Text, 
  Image, 
  ScrollView, 
  ActivityIndicator, 
  Pressable,
  TextInput,
  Dimensions,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import * as ImageManipulator from 'expo-image-manipulator';
import { Button, ButtonText } from '@/components/ui/button';
import FontAwesome5 from '@expo/vector-icons/FontAwesome5';
import FontAwesome6 from '@expo/vector-icons/FontAwesome6';
import Slider from '@react-native-community/slider';
import { DetectionCard } from '@/components/ui/detectionCard';
import {
  segmentFood,
  SegmentationResponse,
  SegmentationDetection,
} from '@/services/segmentationApi';
import { classifyFood } from '@/services/foodApi';
import { 
  searchNutrition, 
  getNutritionDetails, 
  getCachedNutrition 
} from '@/services/nutritionApi';
import { useDetectionStore, ClassifiedDetection, NutritionData } from '@/stores/detectionStore';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function SegmentationScreen() {
  const router = useRouter();
  const { imageUri } = useLocalSearchParams<{ imageUri: string }>();

  const { 
    detections: classifiedDetections, 
    annotatedImage,
    setDetections: setClassifiedDetections,
    setAnnotatedImage,
    updateDetectionName,
    updateDetectionNutrition,
    deleteDetection,
    clearAll,
  } = useDetectionStore();

  const [isProcessing, setIsProcessing] = useState(false);
  const [isClassifying, setIsClassifying] = useState(false);
  const [isFetchingNutrition, setIsFetchingNutrition] = useState(false);
  const [nutritionProgress, setNutritionProgress] = useState({ current: 0, total: 0 });
  const [status, setStatus] = useState('');
  const [prompt, setPrompt] = useState('food');
  const [confidence, setConfidence] = useState(0.5);
  const [result, setResult] = useState<SegmentationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [originalImageSize, setOriginalImageSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    if (imageUri) {
      Image.getSize(
        imageUri,
        (width, height) => setOriginalImageSize({ width, height }),
        (err) => console.error('Failed to get image size:', err)
      );
    }
  }, [imageUri]);

  useEffect(() => {
    if (imageUri && originalImageSize && classifiedDetections.length === 0 && !result) {
      runSegmentation();
    }
  }, [originalImageSize]);

  useEffect(() => {
    if (annotatedImage && !result) {
      setResult({
        annotated_image_base64: annotatedImage,
        num_detections: classifiedDetections.length,
        detections: classifiedDetections,
        message: `Found ${classifiedDetections.length} food items`,
      } as SegmentationResponse);
    }
  }, [annotatedImage]);

  const convertUriToBase64 = async (uri: string) => {
    const manipulated = await ImageManipulator.manipulateAsync(
      uri,
      [{ resize: { width: 640 } }],
      { base64: true, format: ImageManipulator.SaveFormat.JPEG, compress: 0.8 }
    );
    return {
      base64: manipulated.base64!,
      width: manipulated.width,
      height: manipulated.height,
    };
  };

  const cropAndClassify = async (
    detection: SegmentationDetection,
    originalUri: string,
    originalSize: { width: number; height: number },
    segmentedSize: { width: number; height: number }
  ): Promise<ClassifiedDetection> => {
    const { bounding_box: box } = detection;

    try {
      const scaleX = originalSize.width / segmentedSize.width;
      const scaleY = originalSize.height / segmentedSize.height;

      const scaledBox = {
        x1: Math.round(box.x1 * scaleX),
        y1: Math.round(box.y1 * scaleY),
        x2: Math.round(box.x2 * scaleX),
        y2: Math.round(box.y2 * scaleY),
      };

      const padding = 20;
      const cropX = Math.max(0, scaledBox.x1 - padding);
      const cropY = Math.max(0, scaledBox.y1 - padding);
      const cropWidth = Math.min(originalSize.width - cropX, scaledBox.x2 - scaledBox.x1 + padding * 2);
      const cropHeight = Math.min(originalSize.height - cropY, scaledBox.y2 - scaledBox.y1 + padding * 2);

      const cropped = await ImageManipulator.manipulateAsync(
        originalUri,
        [
          { crop: { originX: cropX, originY: cropY, width: cropWidth, height: cropHeight } },
          { resize: { width: 224 } },
        ],
        { base64: true, format: ImageManipulator.SaveFormat.JPEG, compress: 0.9 }
      );

      const base64WithPrefix = `data:image/jpeg;base64,${cropped.base64}`;
      const classificationResult = await classifyFood(base64WithPrefix);

      return {
        ...detection,
        classification: {
          label: classificationResult.top_prediction,
          confidence: classificationResult.predictions[0].confidence,
        },
        croppedImageBase64: `data:image/jpeg;base64,${cropped.base64}`,
        isManuallyEdited: false,
      };
    } catch (err) {
      console.error(`Failed to classify detection ${detection.item_id}:`, err);
      return { ...detection };
    }
  };

  const runSegmentation = async () => {
    if (!imageUri || !originalImageSize || !prompt.trim()) return;

    setIsProcessing(true);
    setError(null);
    setResult(null);
    clearAll();

    try {
      setStatus('Processing image...');
      const { base64, width: segWidth, height: segHeight } = await convertUriToBase64(imageUri);

      setStatus(`Detecting all "${prompt}"...`);
      const segmentationResult = await segmentFood(
        `data:image/jpeg;base64,${base64}`,
        prompt.trim(),
        confidence
      );

      setResult(segmentationResult);
      setAnnotatedImage(segmentationResult.annotated_image_base64 || null);
      setIsProcessing(false);

      if (segmentationResult.detections.length > 0) {
        setIsClassifying(true);
        const classified: ClassifiedDetection[] = [];

        for (let i = 0; i < segmentationResult.detections.length; i++) {
          setStatus(`Classifying item ${i + 1} of ${segmentationResult.detections.length}...`);
          const classifiedDetection = await cropAndClassify(
            segmentationResult.detections[i],
            imageUri,
            originalImageSize,
            { width: segWidth, height: segHeight }
          );
          classified.push(classifiedDetection);
          setClassifiedDetections([...classified]);
        }

        setIsClassifying(false);
        setStatus('');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Segmentation failed');
      setIsProcessing(false);
      setIsClassifying(false);
    }
  };

  const handleUpdateFoodName = (itemId: number, newName: string) => {
    updateDetectionName(itemId, newName);
  };

  const handleDeleteDetection = (itemId: number) => {
    deleteDetection(itemId);
  };

  const handleNutritionPress = (itemId: number, foodName: string) => {
    router.push({ 
      pathname: '/nutrition', 
      params: { 
        foodName,
        itemId: itemId.toString(),
        returnTo: 'segmentation',
        imageUri,
      } 
    });
  };

  // Fetch nutrition for a single detection
  const fetchNutritionForDetection = async (detection: ClassifiedDetection): Promise<NutritionData | null> => {
    const foodName = detection.classification?.label.replace(/_/g, ' ') || '';
    if (!foodName) return null;

    try {
      // Try cache first
      const cached = await getCachedNutrition(foodName);
      if (cached) {
        return {
          name: cached.name,
          default_serving_size: cached.default_serving_size,
          nutrition_per_100g: cached.nutrition_per_100g,
          nutrition_per_serving: cached.nutrition_per_serving,
        };
      }

      // Search and get first result
      const searchResults = await searchNutrition(foodName);
      if (searchResults.results.length > 0) {
        const details = await getNutritionDetails(foodName, 0);
        return {
          name: details.name,
          default_serving_size: details.default_serving_size,
          nutrition_per_100g: details.nutrition_per_100g,
          nutrition_per_serving: details.nutrition_per_serving,
        };
      }
    } catch (err) {
      console.error(`Failed to fetch nutrition for ${foodName}:`, err);
    }
    return null;
  };

  // Handle Estimate Calories button
  const handleEstimateCalories = async () => {
    if (classifiedDetections.length === 0) {
      Alert.alert('No Items', 'Please detect some food items first.');
      return;
    }

    setIsFetchingNutrition(true);
    setNutritionProgress({ current: 0, total: classifiedDetections.length });

    try {
      for (let i = 0; i < classifiedDetections.length; i++) {
        const detection = classifiedDetections[i];
        setNutritionProgress({ current: i + 1, total: classifiedDetections.length });

        // Skip if already has nutrition data
        if (detection.nutrition) continue;

        const nutrition = await fetchNutritionForDetection(detection);
        if (nutrition) {
          updateDetectionNutrition(detection.item_id, nutrition);
        }
      }

      // Navigate to summary page
      router.push('/summary');
    } catch (err) {
      Alert.alert('Error', 'Failed to fetch nutrition data. Please try again.');
    } finally {
      setIsFetchingNutrition(false);
    }
  };

  const totalDetections = result?.num_detections || classifiedDetections.length;

  return (
    <SafeAreaView className="flex-1 bg-white">
      <ScrollView className="flex-1">
        {/* Header */}
        <View className="flex-row items-center justify-between p-4 border-b border-gray-200">
          <Pressable onPress={() => { clearAll(); router.back(); }} className="p-2">
            <FontAwesome5 name="arrow-left" size={20} color="#3b82f6" />
          </Pressable>
          <Text className="text-lg font-bold">Food Segmentation</Text>
          <View style={{ width: 40 }} />
        </View>

        {/* Image Display */}
        <View className="m-4">
          {(result?.annotated_image_base64 || annotatedImage) ? (
            <View className="rounded-xl overflow-hidden shadow-lg">
              <Image
                source={{ uri: result?.annotated_image_base64 || annotatedImage! }}
                style={{ width: SCREEN_WIDTH - 32, height: SCREEN_WIDTH - 32 }}
                resizeMode="contain"
              />
            </View>
          ) : imageUri ? (
            <View className="rounded-xl overflow-hidden shadow-lg">
              <Image
                source={{ uri: imageUri }}
                style={{ width: SCREEN_WIDTH - 32, height: SCREEN_WIDTH - 32 }}
                resizeMode="cover"
              />
            </View>
          ) : null}
        </View>

        {/* Controls */}
        <View className="mx-4 bg-gray-50 rounded-xl p-4 mb-4">
          <Text className="text-sm font-bold text-gray-700 mb-2">Detection Prompt</Text>
          <TextInput
            className="bg-white border border-gray-300 rounded-lg px-4 py-3 text-base"
            placeholder="What to detect (e.g., food, rice, egg)"
            value={prompt}
            onChangeText={setPrompt}
            editable={!isProcessing && !isClassifying && !isFetchingNutrition}
          />

          <Text className="text-sm font-bold text-gray-700 mt-4 mb-2">
            Confidence Threshold: {(confidence * 100).toFixed(0)}%
          </Text>
          <Slider
            style={{ width: '100%', height: 40 }}
            minimumValue={0.1}
            maximumValue={1.0}
            step={0.05}
            value={confidence}
            onValueChange={setConfidence}
            disabled={isProcessing || isClassifying || isFetchingNutrition}
            minimumTrackTintColor="#3b82f6"
            maximumTrackTintColor="#d1d5db"
            thumbTintColor="#3b82f6"
          />

          <Button
            className="mt-4"
            variant="solid"
            size="lg"
            action="primary"
            onPress={runSegmentation}
            disabled={isProcessing || isClassifying || isFetchingNutrition}
          >
            {isProcessing || isClassifying ? (
              <>
                <ActivityIndicator size="small" color="#ffffff" />
                <ButtonText className="ml-2">{status}</ButtonText>
              </>
            ) : (
              <>
                <FontAwesome6 name="wand-magic-sparkles" size={16} color="#ffffff" />
                <ButtonText className="ml-2">{classifiedDetections.length > 0 ? 'Detect Again' : 'Detect & Classify'}</ButtonText>
              </>
            )}
          </Button>
        </View>

        {/* Error */}
        {error && (
          <View className="mx-4 p-4 bg-red-50 rounded-xl mb-4">
            <Text className="text-red-800 font-bold">Error</Text>
            <Text className="text-red-600 mt-1">{error}</Text>
          </View>
        )}

        {/* Results */}
        {(result || classifiedDetections.length > 0) && (
          <View className="mx-4 mb-4">
            {/* Summary */}
            <View className={`rounded-xl p-4 mb-4 ${totalDetections > 0 ? 'bg-green-50' : 'bg-yellow-50'}`}>
              <View className="flex-row items-center">
                <FontAwesome5
                  name={totalDetections > 0 ? 'check-circle' : 'info-circle'}
                  size={20}
                  color={totalDetections > 0 ? '#15803d' : '#a16207'}
                />
                <Text className={`ml-2 font-bold ${totalDetections > 0 ? 'text-green-800' : 'text-yellow-800'}`}>
                  {totalDetections > 0 ? `Found ${totalDetections} food items` : 'No items detected'}
                </Text>
              </View>
              {totalDetections > 0 && (
                <Text className="text-sm text-green-600 mt-1">
                  Showing: {classifiedDetections.length} items
                </Text>
              )}
            </View>

            {/* Detection Cards */}
            {classifiedDetections.length > 0 && (
              <View className="bg-gray-50 rounded-xl p-4">
                <Text className="text-sm font-bold text-gray-700 uppercase tracking-wide mb-1">
                  Detected Foods ({classifiedDetections.length})
                </Text>
                <Text className="text-xs text-gray-500 mb-3">
                  Tap name to edit • Tap 🗑️ to delete • Tap 🍽️ for nutrition
                </Text>

                {classifiedDetections.map((detection) => (
                  <DetectionCard
                    key={detection.item_id}
                    detection={detection}
                    onUpdateName={(name) => handleUpdateFoodName(detection.item_id, name)}
                    onDelete={() => handleDeleteDetection(detection.item_id)}
                    onNutritionPress={() => 
                      handleNutritionPress(
                        detection.item_id, 
                        detection.classification?.label || 'food'
                      )
                    }
                  />
                ))}
              </View>
            )}

            {/* Empty State */}
            {totalDetections > 0 && classifiedDetections.length === 0 && !isClassifying && (
              <View className="bg-gray-50 rounded-xl p-6 items-center">
                <FontAwesome5 name="trash-alt" size={32} color="#9ca3af" />
                <Text className="text-gray-500 mt-3 text-center">All items have been removed.</Text>
                <Button className="mt-4" variant="solid" size="sm" action="secondary" onPress={runSegmentation}>
                  <ButtonText>Detect Again</ButtonText>
                </Button>
              </View>
            )}

            {/* Loading State */}
            {result && result.detections.length > 0 && classifiedDetections.length === 0 && isClassifying && (
              <View className="bg-gray-50 rounded-xl p-4">
                <Text className="text-sm font-bold text-gray-700 uppercase mb-3">Classifying...</Text>
                {result.detections.map((d) => (
                  <View key={d.item_id} className="bg-white rounded-lg p-3 mb-2 border border-gray-200 flex-row items-center">
                    <View className="w-16 h-16 rounded-lg bg-gray-200 items-center justify-center mr-3">
                      <ActivityIndicator size="small" color="#3b82f6" />
                    </View>
                    <Text className="text-gray-500">Classifying item {d.item_id}...</Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}
      </ScrollView>

      {/* Bottom Action */}
      {classifiedDetections.length > 0 && !isClassifying && (
        <View className="p-4 border-t border-gray-200">
          <Button
            variant="solid"
            size="lg"
            action="primary"
            onPress={handleEstimateCalories}
            disabled={isFetchingNutrition}
          >
            {isFetchingNutrition ? (
              <>
                <ActivityIndicator size="small" color="#ffffff" />
                <ButtonText className="ml-2">
                  Fetching nutrition ({nutritionProgress.current}/{nutritionProgress.total})...
                </ButtonText>
              </>
            ) : (
              <>
                <FontAwesome6 name="fire-flame-curved" size={16} color="#ffffff" style={{ marginRight: 8 }} />
                <ButtonText>Estimate Calories</ButtonText>
              </>
            )}
          </Button>
        </View>
      )}
    </SafeAreaView>
  );
}