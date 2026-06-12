// app/nutrition.tsx
import { View, Text, ScrollView, ActivityIndicator, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Button, ButtonText } from '@/components/ui/button';
import FontAwesome5 from '@expo/vector-icons/FontAwesome5';
import FontAwesome6 from '@expo/vector-icons/FontAwesome6';
import {
  searchNutrition,
  getNutritionDetails,
  NutritionSearchResult,
  NutritionDetails,
  getCachedNutrition,
  sortNutrients,
  sortAdditionalInfo
} from '@/services/nutritionApi';
import { useDetectionStore } from '../stores/detectionStore';

export default function NutritionScreen() {
  const router = useRouter();
  const { 
    foodName, 
    itemId,
    returnTo,
    imageUri,
  } = useLocalSearchParams<{ 
    foodName: string;
    itemId?: string;
    returnTo?: string;
    imageUri?: string;
  }>();

  const [isSearching, setIsSearching] = useState(true);
  const [isLoadingDetails, setIsLoadingDetails] = useState(false);
  const [searchResults, setSearchResults] = useState<NutritionSearchResult[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [nutritionDetails, setNutritionDetails] = useState<NutritionDetails | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (foodName) {
      handleSearch();
    }
  }, [foodName]);

  const handleSearch = async () => {
    if (!foodName) return;

    setIsSearching(true);
    setError(null);
    setSearchResults([]);
    setNutritionDetails(null);
    setSelectedIndex(null);

    try {
      const response = await searchNutrition(foodName.replace(/_/g, ' '));
      setSearchResults(response.results);

      if (response.results.length > 0) {
        handleSelectResult(0, response.results);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setIsSearching(false);
    }
  };

  const handleSelectResult = async (
    index: number,
    results: NutritionSearchResult[] = searchResults
  ) => {
    if (!foodName) return;
    setSelectedIndex(index);
    setIsLoadingDetails(true);
    setError(null);
    
    try {
      const selectedFood = results[index];
      const cached = await getCachedNutrition(selectedFood.name);
      
      if (cached) {
        setNutritionDetails(cached);
        return;
      }
      
      const details = await getNutritionDetails(foodName.replace(/_/g, ' '), index);
      setNutritionDetails(details);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load details');
    } finally {
      setIsLoadingDetails(false);
    }
  };

  // Handle "Select This" button
  const handleSelectFood = () => {
    if (returnTo === 'segmentation' && itemId && nutritionDetails) {
      // Update the store directly, then go back
      const { updateDetectionName } = useDetectionStore.getState();
      updateDetectionName(parseInt(itemId, 10), nutritionDetails.name);
      router.back();
    } else {
      router.push('/');
    }
  };

  return (
    <SafeAreaView className="flex-1 bg-white">
      <ScrollView className="flex-1">
        {/* Header */}
        <View className="flex-row items-center justify-between p-4 border-b border-gray-200">
          <Pressable onPress={() => router.back()} className="p-2">
            <FontAwesome5 name="arrow-left" size={20} color="#3b82f6" />
          </Pressable>
          <Text className="text-lg font-bold">Nutrition Info</Text>
          <View style={{ width: 40 }} />
        </View>

        {/* Search Query Display */}
        <View className="mx-4 mt-4 bg-blue-50 rounded-xl p-4">
          <Text className="text-sm text-blue-600">Searching for:</Text>
          <Text className="text-lg font-bold text-blue-800">
            {foodName?.replace(/_/g, ' ')}
          </Text>
        </View>

        {/* Loading State */}
        {isSearching && (
          <View className="items-center py-8">
            <ActivityIndicator size="large" color="#3b82f6" />
            <Text className="mt-4 text-gray-600">Searching nutrition database...</Text>
            <Text className="mt-1 text-xs text-gray-400">This may take a few seconds</Text>
          </View>
        )}

        {/* Error State */}
        {error && !isSearching && (
          <View className="mx-4 mt-4 p-4 bg-red-50 rounded-xl">
            <Text className="text-red-800 font-bold">Error</Text>
            <Text className="text-red-600 mt-1">{error}</Text>
            <Button className="mt-4" variant="solid" action="negative" onPress={handleSearch}>
              <ButtonText>Try Again</ButtonText>
            </Button>
          </View>
        )}

        {/* Search Results Selector */}
        {searchResults.length > 0 && !isSearching && (
          <View className="mx-4 mt-4">
            <Text className="text-sm text-gray-600 mb-2">
              Found {searchResults.length} results - tap to select:
            </Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View className="flex-row gap-2">
                {searchResults.map((result) => (
                  <Pressable
                    key={result.index}
                    onPress={() => handleSelectResult(result.index)}
                    className={`px-4 py-2 rounded-full border ${
                      selectedIndex === result.index
                        ? 'bg-blue-500 border-blue-500'
                        : 'bg-white border-gray-300'
                    }`}
                  >
                    <Text
                      className={`text-sm ${
                        selectedIndex === result.index ? 'text-white font-medium' : 'text-gray-700'
                      }`}
                      numberOfLines={1}
                    >
                      {result.name}
                    </Text>
                  </Pressable>
                ))}
              </View>
            </ScrollView>
          </View>
        )}

        {/* Loading Details */}
        {isLoadingDetails && (
          <View className="items-center py-8">
            <ActivityIndicator size="large" color="#22c55e" />
            <Text className="mt-4 text-gray-600">Loading nutrition details...</Text>
          </View>
        )}

        {/* Nutrition Details */}
        {nutritionDetails && !isLoadingDetails && (
          <View className="mx-4 mt-4">
            {/* Food Name & Description */}
            <View className="bg-green-50 rounded-xl p-4 mb-4">
              <Text className="text-xl font-bold text-green-800">
                {nutritionDetails.name}
              </Text>
              {nutritionDetails.description && (
                <Text className="text-sm text-green-600 mt-2">
                  {nutritionDetails.description}
                </Text>
              )}
              {nutritionDetails.default_serving_size && (
                <View className="flex-row items-center mt-3">
                  <FontAwesome6 name="utensils" size={14} color="#15803d" />
                  <Text className="text-sm text-green-700 ml-2">
                    Serving: {nutritionDetails.default_serving_size}
                  </Text>
                </View>
              )}
            </View>

            {/* Nutrition Per 100g */}
            <View className="bg-gray-50 rounded-xl p-4 mb-4">
              <Text className="text-sm font-bold text-gray-700 uppercase tracking-wide mb-3">
                Nutrition per 100g
              </Text>
              {sortNutrients(nutritionDetails.nutrition_per_100g).map(([nutrient, value]) => (
                <View
                  key={nutrient}
                  className="flex-row justify-between py-2 border-b border-gray-200"
                >
                  <Text className="text-gray-700">{nutrient}</Text>
                  <Text className="text-gray-900 font-medium">{value}</Text>
                </View>
              ))}
            </View>

            {/* Nutrition Per Serving */}
            {nutritionDetails.nutrition_per_serving && (
              <View className="bg-orange-50 rounded-xl p-4 mb-4">
                <Text className="text-sm font-bold text-orange-700 uppercase tracking-wide mb-3">
                  Per Serving ({nutritionDetails.default_serving_size})
                </Text>
                {sortNutrients(nutritionDetails.nutrition_per_serving).map(
                  ([nutrient, value]) => (
                    <View
                      key={nutrient}
                      className="flex-row justify-between py-2 border-b border-orange-200"
                    >
                      <Text className="text-orange-700">{nutrient}</Text>
                      <Text className="text-orange-900 font-bold">{value}</Text>
                    </View>
                  )
                )}
              </View>
            )}

            {/* Extra Info */}
            {nutritionDetails.extra_info && Object.keys(nutritionDetails.extra_info).length > 0 && (
              <View className="bg-purple-50 rounded-xl p-4 mb-4">
                <Text className="text-sm font-bold text-purple-700 uppercase tracking-wide mb-3">
                  Additional Info
                </Text>
                {sortAdditionalInfo(nutritionDetails.extra_info).map(([key, value]) => (
                  <View key={key} className="flex-row justify-between py-2 border-b border-purple-200">
                    <Text className="text-purple-700 flex-1">{key}</Text>
                    <Text className="text-purple-900 font-medium flex-1 text-right">{value}</Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}

        {/* No Results */}
        {!isSearching && searchResults.length === 0 && !error && (
          <View className="mx-4 mt-8 items-center">
            <FontAwesome5 name="search" size={48} color="#d1d5db" />
            <Text className="text-gray-500 mt-4 text-center">
              No nutrition information found for this food.
            </Text>
            <Button className="mt-4" variant="solid" action="secondary" onPress={() => router.back()}>
              <ButtonText>Go Back</ButtonText>
            </Button>
          </View>
        )}
      </ScrollView>

      {/* Bottom Action */}
      {nutritionDetails && (
        <View className="p-4 border-t border-gray-200">
          <Button
            variant="solid"
            size="lg"
            action="primary"
            onPress={handleSelectFood}
          >
            <FontAwesome5 name="check" size={16} color="#ffffff" style={{ marginRight: 8 }} />
            <ButtonText>
              {returnTo === 'segmentation' ? 'Select This Food' : 'Done'}
            </ButtonText>
          </Button>
        </View>
      )}
    </SafeAreaView>
  );
}