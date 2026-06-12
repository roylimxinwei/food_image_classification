// components/ui/detectionCard.tsx
import { 
  View, 
  Text, 
  Image, 
  Pressable, 
  Modal, 
  TouchableOpacity, 
  TextInput,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { useState } from 'react';
import FontAwesome5 from '@expo/vector-icons/FontAwesome5';
import { Button, ButtonText } from '@/components/ui/button';

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

interface ClassifiedDetection {
  item_id: number;
  confidence: number;
  bounding_box: BoundingBox;
  classification?: Classification;
  croppedImageBase64?: string;
  isManuallyEdited?: boolean;
}

interface DetectionCardProps {
  detection: ClassifiedDetection;
  onUpdateName: (name: string) => void;
  onDelete: () => void;
  onNutritionPress: () => void;  // No parameters - segmentation handles it via closure
}

export function DetectionCard({
  detection,
  onUpdateName,
  onDelete,
  onNutritionPress,
}: DetectionCardProps) {
  const { classification, croppedImageBase64, isManuallyEdited } = detection;
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const [editedName, setEditedName] = useState('');

  const foodLabel = classification?.label.replace(/_/g, ' ') || 'Unknown';
  const classConfidence = classification ? (classification.confidence * 100).toFixed(1) : null;
  const segConfidence = (detection.confidence * 100).toFixed(1);

  const openEdit = () => {
    setEditedName(foodLabel);
    setIsEditModalVisible(true);
  };

  const saveEdit = () => {
    if (editedName.trim()) onUpdateName(editedName.trim());
    setIsEditModalVisible(false);
  };

  const confirmDelete = () => {
    Alert.alert(
      'Delete Item',
      `Are you sure you want to remove "${foodLabel}"?`,
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Delete', style: 'destructive', onPress: onDelete },
      ]
    );
  };

  return (
    <>
      <View className="p-3 border border-gray-200 rounded-lg bg-white mb-3">
        <View className="flex-row">
          {/* Thumbnail */}
          <View className="mr-3">
            {croppedImageBase64 ? (
              <Image 
                source={{ uri: croppedImageBase64 }} 
                className="w-16 h-16 rounded-lg" 
                resizeMode="cover" 
              />
            ) : (
              <View className="w-16 h-16 rounded-lg bg-gray-200 items-center justify-center">
                <ActivityIndicator size="small" color="#3b82f6" />
              </View>
            )}
          </View>

          {/* Info */}
          <View className="flex-1 justify-center">
            <Pressable onPress={openEdit} className="flex-row items-center">
              <Text 
                className="text-base font-bold text-blue-600 capitalize underline" 
                numberOfLines={2}
              >
                {foodLabel}
              </Text>
              <FontAwesome5 name="pencil-alt" size={12} color="#3b82f6" style={{ marginLeft: 6 }} />
            </Pressable>

            <View className="flex-row flex-wrap gap-1 mt-1">
              {isManuallyEdited ? (
                <View className="bg-purple-100 rounded-full px-2 py-0.5">
                  <Text className="text-xs text-purple-800">Edited</Text>
                </View>
              ) : classConfidence ? (
                <View className="bg-green-100 rounded-full px-2 py-0.5">
                  <Text className="text-xs text-green-800">Class: {classConfidence}%</Text>
                </View>
              ) : null}
              <View className="bg-blue-100 rounded-full px-2 py-0.5">
                <Text className="text-xs text-blue-800">Seg: {segConfidence}%</Text>
              </View>
            </View>
          </View>

          {/* Action Buttons */}
          <View className="flex-row items-center gap-2">
            {/* Nutrition Button */}
            {classification && (
              <Pressable
                className="bg-orange-50 rounded-lg px-2 py-2 justify-center items-center"
                onPress={onNutritionPress}
              >
                <FontAwesome5 name="utensils" size={16} color="#ea580c" />
              </Pressable>
            )}

            {/* Delete Button */}
            <Pressable
              className="bg-red-50 rounded-lg px-2 py-2 justify-center items-center"
              onPress={confirmDelete}
            >
              <FontAwesome5 name="trash-alt" size={16} color="#ef4444" />
            </Pressable>
          </View>
        </View>
      </View>

      {/* Edit Modal */}
      <Modal 
        visible={isEditModalVisible} 
        transparent 
        animationType="fade" 
        onRequestClose={() => setIsEditModalVisible(false)}
      >
        <KeyboardAvoidingView 
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'} 
          className="flex-1"
        >
          <TouchableOpacity
            className="flex-1 bg-black/50 justify-center items-center"
            activeOpacity={1}
            onPress={() => setIsEditModalVisible(false)}
          >
            <TouchableOpacity 
              activeOpacity={1} 
              className="bg-white rounded-xl p-5 mx-6 w-11/12 max-w-sm"
            >
              <View className="flex-row items-center mb-4">
                <FontAwesome5 name="edit" size={18} color="#3b82f6" />
                <Text className="font-bold text-lg ml-2">Edit Food Name</Text>
              </View>

              {croppedImageBase64 && (
                <View className="items-center mb-4">
                  <Image 
                    source={{ uri: croppedImageBase64 }} 
                    className="w-24 h-24 rounded-lg" 
                    resizeMode="cover" 
                  />
                </View>
              )}

              <Text className="text-sm text-gray-600 mb-2">Enter the correct food name:</Text>
              <TextInput
                className="bg-gray-50 border border-gray-300 rounded-lg px-4 py-3 text-base min-h-14"
                placeholder="e.g., Chicken Rice"
                value={editedName}
                onChangeText={setEditedName}
                autoFocus
                autoCapitalize="words"
              />

              <View className="flex-row gap-3 mt-5">
                <View className="flex-1">
                  <Button 
                    variant="solid" 
                    size="md" 
                    action="secondary" 
                    onPress={() => setIsEditModalVisible(false)}
                  >
                    <ButtonText>Cancel</ButtonText>
                  </Button>
                </View>
                <View className="flex-1">
                  <Button 
                    variant="solid" 
                    size="md" 
                    action="primary" 
                    onPress={saveEdit}
                  >
                    <ButtonText>Save</ButtonText>
                  </Button>
                </View>
              </View>
            </TouchableOpacity>
          </TouchableOpacity>
        </KeyboardAvoidingView>
      </Modal>
    </>
  );
}