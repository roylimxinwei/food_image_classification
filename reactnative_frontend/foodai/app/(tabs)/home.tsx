import { Text, View, Alert, Image, Pressable, Modal, TouchableOpacity, ActivityIndicator } from "react-native";
import Animated, { 
  useSharedValue, 
  useAnimatedStyle, 
  withRepeat, 
  withSequence, 
  withTiming,
  Easing,
  cancelAnimation
} from 'react-native-reanimated';
import {
  Button,
  ButtonText,
} from '@/components/ui/button';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { useState, useEffect } from 'react';
import FontAwesome5 from '@expo/vector-icons/FontAwesome5';
import FontAwesome from '@expo/vector-icons/FontAwesome';
import FontAwesome6 from '@expo/vector-icons/FontAwesome6';
import {checkApiHealth } from '@/services/foodApi';
import { useRouter } from 'expo-router';

export default function HomeTab() {
  const router = useRouter();

  const [image, setImage] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);

  const translateY = useSharedValue(0);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    checkApiHealth().then(setApiConnected);
  }, []);

  useEffect(() => {
    // Start bounce animation loop with reanimated
    translateY.value = withRepeat(
      withSequence(
        withTiming(-10, { duration: 1000, easing: Easing.inOut(Easing.ease) }),
        withTiming(0, { duration: 1000, easing: Easing.inOut(Easing.ease) })
      ),
      -1, // -1 means infinite
      false
    );
    return () => {
    cancelAnimation(translateY);
    translateY.value = 0;
  };
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
  }));

  const takePhoto = async () => {
    // Request camera permission
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Camera permission is required to take photos');
      return;
    }

    // Launch camera
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const pickImage = async () => {
    // Request gallery permission
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Gallery permission is required to select photos');
      return;
    }

    // Launch image picker
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };


  const goToAnalysis = () => {
    if (!image) {
      Alert.alert('No Image', 'Please take or select an image first');
      return;
    }

    // Navigate to analysis screen with image data
    router.push({
      pathname: '/analysis',
      params: {
        imageUri: image,
      },
    });
  };

  const clearImage = () => {
    setImage(null);
    // setImageBase64(null);
    // setResult(null);
  };

  const handleOpen = () => {
    setIsOpen(true);
  };
  const handleClose = () => {
    setIsOpen(false);
  };

  return (
    <SafeAreaView className="flex-1">
      <View className="flex-1 justify-between items-center p-8">
        {/* API Connection Status (for debugging) */}
        <View className="absolute top-2 right-2">
          <View className={`w-3 h-3 rounded-full ${
            apiConnected === null ? 'bg-gray-400' : 
            apiConnected ? 'bg-green-500' : 'bg-red-500'
          }`} />
        </View>
        <View className="items-center mt-16">
          <Text className="text-2xl font-bold">Snap Your Meal</Text>
          <Text className="text-base">Let AI Analyse Your Meal Instantly</Text>
        </View>
        
        {image && (
          <View className="w-full aspect-square rounded-lg overflow-hidden">
            <Image source={{ uri: image }} className="w-full h-full" resizeMode="cover" />
          </View>
        )}
      

        {image ? (
          <View className="w-full">
            <Button variant="solid" size="lg" action="negative" onPress={clearImage}>
              <ButtonText>Clear Image</ButtonText>
            </Button>
          </View>
        ): (<View className="w-full aspect-square rounded-lg border-2 border-dashed border-gray-300 justify-center items-center">
            <Pressable onPress={takePhoto}>
              <Animated.View 
                style={animatedStyle}
                className="p-6 mb-4 aspect-square rounded-full border-2 border-gray-300 justify-center items-center"
              >
                <FontAwesome5 name="camera" size={54} color="#3b82f6" />
              </Animated.View>
            </Pressable>
            <Text className="p-4 text-gray-400 text-center px-4">Take a photo of your meal to automatically track calories and nutrients</Text>
            <Button className="p-4" variant="solid" size="md" action="positive" onPress={handleOpen}>
              <ButtonText>AI Powered</ButtonText>
            </Button>
            <Modal
              visible={isOpen}
              transparent
              animationType="fade"
              onRequestClose={handleClose}
            >
              <TouchableOpacity 
                className="flex-1 bg-black/50 justify-center items-center"
                activeOpacity={1}
                onPress={handleClose}
              >
                <TouchableOpacity activeOpacity={1} className="bg-white rounded-xl p-6 mx-8 shadow-lg">
                  <Text className="font-bold text-lg mb-2">Nutrition Analysis</Text>
                  <Text className="text-gray-600">
                    Take a photo of your meal to get instant nutritional information!
                  </Text>
                  <Button className="mt-4" size="sm" onPress={handleClose}>
                    <ButtonText>Got it!</ButtonText>
                  </Button>
                </TouchableOpacity>
              </TouchableOpacity>
            </Modal>
          </View>)}

        <View className="flex-row gap-4 w-full">
          <View className="flex-1">
            <Button className="flex-row" variant="solid" size="lg" action="secondary" onPress={pickImage}>
              <FontAwesome className="mr-2" name="picture-o" size={18} color="#ffffff" />
              <ButtonText>Gallery</ButtonText>
            </Button>
          </View>
          <View className="flex-1">
            <Button className="flex-row" variant="solid" size="lg" action="primary" onPress={takePhoto}>
              <FontAwesome5 className="mr-2" name="camera" size={18} color="#ffffff" />
              <ButtonText>{image ? 'Retake' : 'Take Photo'} </ButtonText>
            </Button>
          </View>
        </View>
      </View>
      <View className="px-8">
          {image ? (
            <Button 
              className="flex-row w-full" 
              variant="solid" 
              size="lg" 
              action="positive"
              onPress={goToAnalysis}
              >
                <FontAwesome6 className="mr-2" name="bolt-lightning" size={18} color="#ffffff" />
                <ButtonText>Analyse with AI</ButtonText>
              </Button>
            ) : (
              <Button className="flex-row w-full" variant="solid" size="lg" action="secondary" onPress={handleOpen}>
                <FontAwesome6 className="mr-2" name="bolt-lightning" size={18} color="#ffffff" />
                <ButtonText>Analyse with AI</ButtonText>
              </Button>
            )}
        </View>
    </SafeAreaView>
  );
}


