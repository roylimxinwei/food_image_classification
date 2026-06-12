import { View, Text, Image, ScrollView, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import * as FileSystem from 'expo-file-system';
import { Button, ButtonText } from '@/components/ui/button';
import FontAwesome5 from '@expo/vector-icons/FontAwesome5';
import { classifyFood, ClassificationResponse } from '@/services/foodApi';
import FontAwesome6 from '@expo/vector-icons/FontAwesome6';

export default function AnalysisTab() {
    const router = useRouter();
    const { imageUri } = useLocalSearchParams<{ imageUri: string }>();

    const [isAnalyzing, setIsAnalyzing] = useState(true);
    const [status, setStatus] = useState('Converting image...');
    const [result, setResult] = useState<ClassificationResponse | null>(null);
    const [error, setError] = useState<string | null>(null);


    useEffect(() => {
        if (imageUri) {
            analyzeImage();
        }
    }, [imageUri])

    const convertUriToBase64 = async (uri: string) => {
        try {
            const file = new FileSystem.File(uri);
            const base64 = await file.base64();
            return base64;
        } catch (error) {
            throw new Error('Failed to convert image to base 64')
        }
    }

    const analyzeImage = async () => {
        if (!imageUri) {
        setError('No image provided');
        setIsAnalyzing(false);
        return;
        }

        setIsAnalyzing(true);
        setError(null);

        try {
        // Step 1: Convert URI to base64
        setStatus('Converting image...');
        const base64 = await convertUriToBase64(imageUri);
        
        // Step 2: Call API
        setStatus('Analyzing with AI...');
        const base64WithPrefix = `data:image/jpeg;base64,${base64}`;
        const response = await classifyFood(base64WithPrefix);
        
        setResult(response);
        } catch (err) {
        setError(err instanceof Error ? err.message : 'Analysis failed');
        } finally {
        setIsAnalyzing(false);
        }
    };

    return (
    <SafeAreaView className="flex-1">
      <ScrollView className="flex-1">
        {/* Header */}
        <View className="flex-row items-center justify-between p-4 border-b border-gray-200">
          <Button variant="ghost" onPress={() => router.back()}>
            <FontAwesome5 name="arrow-left" size={20} color="#3b82f6" />
          </Button>
          <Text className="text-xl font-bold">Food Analysis</Text>
          <View style={{ width: 40 }} />
        </View>
        {/* Image Preview */}
        {imageUri && (
          <View className="m-4 rounded-xl overflow-hidden shadow-lg">
            <Image 
              source={{ uri: imageUri }} 
              className="w-full aspect-square"
              resizeMode="cover"
            />
          </View>
        )}

        {/* Loading State */}
        {isAnalyzing && (
          <View className="items-center py-8">
            <ActivityIndicator size="large" color="#3b82f6" />
            <Text className="mt-4 text-gray-600">{status}</Text>
          </View>
        )}

        {/* Error State */}
        {error && (
          <View className="mx-4 p-4 bg-red-50 rounded-xl">
            <Text className="text-red-800 font-bold">Analysis Failed</Text>
            <Text className="text-red-600 mt-1">{error}</Text>
            <Button 
              className="mt-4" 
              variant="solid" 
              action="negative"
              onPress={analyzeImage}
            >
              <ButtonText>Try Again</ButtonText>
            </Button>
          </View>
        )}

        {/* Results */}
        {result && (
          <View className="mx-4">
            {/* Top Prediction Card */}
            <View className="bg-green-50 rounded-xl p-6 mb-4">
              <Text className="text-sm text-green-600 uppercase tracking-wide">
                Detected Food
              </Text>
              <Text className="text-2xl font-bold text-green-800 mt-1">
                {result.top_prediction.replace(/_/g, ' ')}
              </Text>
              <View className="flex-row items-center mt-2">
                <View className="bg-green-200 rounded-full px-3 py-1">
                  <Text className="text-green-800 font-medium">
                    {(result.predictions[0].confidence * 100).toFixed(1)}% confident
                  </Text>
                </View>
              </View>
            </View>

            {/* All Predictions */}
            <View className="bg-gray-50 rounded-xl p-4 mb-4">
              <Text className="text-sm text-gray-600 uppercase tracking-wide mb-3">
                Other Possibilities
              </Text>
              {result.predictions.slice(1).map((pred) => (
                <View 
                  key={pred.rank} 
                  className="flex-row justify-between items-center py-2 border-b border-gray-200"
                >
                  <Text className="text-gray-800">
                    {pred.label.replace(/_/g, ' ')}
                  </Text>
                  <Text className="text-gray-500">
                    {(pred.confidence * 100).toFixed(1)}%
                  </Text>
                </View>
              ))}
            </View>

            {/* Model Info */}
            <View className="bg-blue-50 rounded-xl p-4 mb-4">
              <Text className="text-xs text-blue-600">
                Model: {result.model_name}
              </Text>
            </View>
          </View>
        )}
      </ScrollView>

      {/* Bottom Actions */}
      {result && (
        <View className="p-4 border-t border-gray-200 gap-3">
          {/* Segmentation Button */}
          <Button
            variant="solid"
            size="lg"
            action="primary"
            onPress={() => {
              router.push({
                pathname: '/segmentation',
                params: { 
                  imageUri: imageUri,
                  // foodName: result.top_prediction 
                }
              });
            }}
          >
            <FontAwesome6 className="mb-1" name="wand-magic-sparkles" size={16} color="#ffffff" />
            <ButtonText className="ml-2">Segment Food Items</ButtonText>
          </Button>

          {/* Nutrition Button */}
          {/* <Button
            variant="solid"
            size="lg"
            action="primary"
            onPress={() => {
              router.push({
                pathname: '/nutrition',
                params: { foodName: result.top_prediction }
              });
            }}
          >
            <FontAwesome5 name="search" size={16} color="#ffffff" />
            <ButtonText className="ml-2">Get Nutrition Info</ButtonText>
          </Button> */}
        </View>
      )}
    </SafeAreaView>
  );
}