// app/summary.tsx
import { 
  View, 
  Text, 
  ScrollView, 
  Image, 
  Pressable,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useMemo } from 'react';
import { Button, ButtonText } from '@/components/ui/button';
import FontAwesome5 from '@expo/vector-icons/FontAwesome5';
import FontAwesome6 from '@expo/vector-icons/FontAwesome6';
import { useDetectionStore } from '@/stores/detectionStore';

// Nutrient display order
const NUTRIENT_ORDER = [
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

// Key nutrients to highlight
const KEY_NUTRIENTS = [
  "Energy (kcal)",
  "Protein (g)",
  "Carbohydrate (g)",
  "Total Fat (g)",
  "Dietary Fibre (g)",
];

export default function SummaryScreen() {
  const router = useRouter();
  const { detections, clearAll } = useDetectionStore();

  // Calculate totals from all detections
  const { totals, itemsWithNutrition, itemsWithoutNutrition } = useMemo(() => {
    const totals: Record<string, number> = {};
    const withNutrition: typeof detections = [];
    const withoutNutrition: typeof detections = [];

    detections.forEach((detection) => {
      if (detection.nutrition?.nutrition_per_serving) {
        withNutrition.push(detection);
        
        Object.entries(detection.nutrition.nutrition_per_serving).forEach(([nutrient, value]) => {
          if (typeof value === 'number' && !isNaN(value)) {
            totals[nutrient] = (totals[nutrient] || 0) + value;
          }
        });
      } else {
        withoutNutrition.push(detection);
      }
    });

    // Round totals
    Object.keys(totals).forEach((key) => {
      totals[key] = Math.round(totals[key] * 10) / 10;
    });

    return { totals, itemsWithNutrition: withNutrition, itemsWithoutNutrition: withoutNutrition };
  }, [detections]);

  // Sort totals by defined order
  const sortedTotals = useMemo(() => {
    return Object.entries(totals).sort((a, b) => {
      const indexA = NUTRIENT_ORDER.indexOf(a[0]);
      const indexB = NUTRIENT_ORDER.indexOf(b[0]);
      const orderA = indexA === -1 ? 999 : indexA;
      const orderB = indexB === -1 ? 999 : indexB;
      return orderA - orderB;
    });
  }, [totals]);

  // Get just key nutrients for the summary card
  const keyNutrientTotals = useMemo(() => {
    return sortedTotals.filter(([nutrient]) => KEY_NUTRIENTS.includes(nutrient));
  }, [sortedTotals]);

  const handleDone = () => {
    clearAll();
    router.push('/');
  };

  const totalCalories = totals["Energy (kcal)"] || 0;

  return (
    <SafeAreaView className="flex-1 bg-white">
      <ScrollView className="flex-1">
        {/* Header */}
        <View className="flex-row items-center justify-between p-4 border-b border-gray-200">
          <Pressable onPress={() => router.back()} className="p-2">
            <FontAwesome5 name="arrow-left" size={20} color="#3b82f6" />
          </Pressable>
          <Text className="text-lg font-bold">Meal Summary</Text>
          <View style={{ width: 40 }} />
        </View>

        {/* Total Calories Hero */}
        <View className="mx-4 mt-4 bg-gradient-to-br from-green-500 to-green-600 rounded-2xl p-6 bg-green-500">
          <View className="items-center">
            <FontAwesome6 name="fire-flame-curved" size={32} color="#ffffff" />
            <Text className="text-white text-lg mt-2">Total Calories</Text>
            <Text className="text-white text-5xl font-bold mt-1">
              {totalCalories.toFixed(0)}
            </Text>
            <Text className="text-green-100 text-sm mt-1">kcal</Text>
          </View>

          {/* Key Macros Row */}
          <View className="flex-row justify-around mt-6 pt-4 border-t border-green-400">
            {keyNutrientTotals
              .filter(([n]) => n !== "Energy (kcal)")
              .slice(0, 4)
              .map(([nutrient, value]) => {
                const unit = nutrient.match(/\(([^)]+)\)/)?.[1] || '';
                const name = nutrient.replace(/\s*\([^)]*\)/, '');
                return (
                  <View key={nutrient} className="items-center">
                    <Text className="text-green-100 text-xs">{name}</Text>
                    <Text className="text-white text-lg font-bold">{value}</Text>
                    <Text className="text-green-200 text-xs">{unit}</Text>
                  </View>
                );
              })}
          </View>
        </View>

        {/* Items Breakdown */}
        <View className="mx-4 mt-6">
          <Text className="text-sm font-bold text-gray-700 uppercase tracking-wide mb-3">
            Food Items ({itemsWithNutrition.length})
          </Text>

          {itemsWithNutrition.map((detection) => (
            <View
              key={detection.item_id}
              className="bg-gray-50 rounded-xl p-4 mb-3"
            >
              <View className="flex-row">
                {/* Thumbnail */}
                {detection.croppedImageBase64 && (
                  <Image
                    source={{ uri: detection.croppedImageBase64 }}
                    className="w-16 h-16 rounded-lg mr-3"
                    resizeMode="cover"
                  />
                )}

                {/* Info */}
                <View className="flex-1">
                  <Text className="text-base font-bold text-gray-800 capitalize">
                    {detection.nutrition?.name || detection.classification?.label.replace(/_/g, ' ')}
                  </Text>
                  {detection.nutrition?.default_serving_size && (
                    <Text className="text-xs text-gray-500 mt-0.5">
                      {detection.nutrition.default_serving_size}
                    </Text>
                  )}

                  {/* Per-item calories */}
                  <View className="flex-row items-center mt-2">
                    <FontAwesome6 name="fire-flame-curved" size={12} color="#f97316" />
                    <Text className="text-orange-600 font-bold ml-1">
                      {detection.nutrition?.nutrition_per_serving?.["Energy (kcal)"]?.toFixed(0) || '-'} kcal
                    </Text>
                  </View>
                </View>

                {/* Quick Stats */}
                <View className="items-end">
                  <View className="bg-orange-100 rounded-lg px-2 py-1">
                    <Text className="text-xs text-orange-700">
                      P: {detection.nutrition?.nutrition_per_serving?.["Protein (g)"]?.toFixed(1) || '-'}g
                    </Text>
                  </View>
                  <View className="bg-blue-100 rounded-lg px-2 py-1 mt-1">
                    <Text className="text-xs text-blue-700">
                      C: {detection.nutrition?.nutrition_per_serving?.["Carbohydrate (g)"]?.toFixed(1) || '-'}g
                    </Text>
                  </View>
                  <View className="bg-yellow-100 rounded-lg px-2 py-1 mt-1">
                    <Text className="text-xs text-yellow-700">
                      F: {detection.nutrition?.nutrition_per_serving?.["Total Fat (g)"]?.toFixed(1) || '-'}g
                    </Text>
                  </View>
                </View>
              </View>
            </View>
          ))}

          {/* Items without nutrition */}
          {itemsWithoutNutrition.length > 0 && (
            <View className="bg-yellow-50 rounded-xl p-4 mt-2">
              <View className="flex-row items-center mb-2">
                <FontAwesome5 name="exclamation-triangle" size={14} color="#a16207" />
                <Text className="text-yellow-800 font-bold ml-2">
                  Missing Nutrition Data ({itemsWithoutNutrition.length})
                </Text>
              </View>
              {itemsWithoutNutrition.map((detection) => (
                <Text key={detection.item_id} className="text-yellow-700 text-sm capitalize">
                  • {detection.classification?.label.replace(/_/g, ' ') || 'Unknown'}
                </Text>
              ))}
              <Text className="text-yellow-600 text-xs mt-2">
                Tap back and select nutrition info for these items
              </Text>
            </View>
          )}
        </View>

        {/* Full Nutrition Breakdown */}
        {sortedTotals.length > 0 && (
          <View className="mx-4 mt-6 mb-4">
            <Text className="text-sm font-bold text-gray-700 uppercase tracking-wide mb-3">
              Complete Nutrition Breakdown
            </Text>
            <View className="bg-gray-50 rounded-xl p-4">
              {sortedTotals.map(([nutrient, value]) => {
                const isKey = KEY_NUTRIENTS.includes(nutrient);
                return (
                  <View
                    key={nutrient}
                    className={`flex-row justify-between py-2 border-b border-gray-200 ${isKey ? 'bg-green-50 -mx-2 px-2 rounded' : ''}`}
                  >
                    <Text className={`${isKey ? 'font-bold text-green-800' : 'text-gray-700'}`}>
                      {nutrient}
                    </Text>
                    <Text className={`${isKey ? 'font-bold text-green-800' : 'text-gray-900 font-medium'}`}>
                      {value}
                    </Text>
                  </View>
                );
              })}
            </View>
          </View>
        )}

        {/* Empty State */}
        {detections.length === 0 && (
          <View className="mx-4 mt-8 items-center">
            <FontAwesome5 name="utensils" size={48} color="#d1d5db" />
            <Text className="text-gray-500 mt-4 text-center">
              No food items to summarize.
            </Text>
            <Button className="mt-4" variant="solid" action="secondary" onPress={() => router.push('/')}>
              <ButtonText>Go Home</ButtonText>
            </Button>
          </View>
        )}
      </ScrollView>

      {/* Bottom Action */}
      {detections.length > 0 && (
        <View className="p-4 border-t border-gray-200">
          <Button
            variant="solid"
            size="lg"
            action="primary"
            onPress={handleDone}
          >
            <FontAwesome5 name="check" size={16} color="#ffffff" style={{ marginRight: 8 }} />
            <ButtonText>Done</ButtonText>
          </Button>
        </View>
      )}
    </SafeAreaView>
  );
}