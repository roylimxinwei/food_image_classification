import React, { useRef } from 'react';
import {
  View,
  Animated,
  PanResponder,
  Pressable,
  Text,
  Dimensions,
  StyleSheet,
} from 'react-native';
import FontAwesome5 from '@expo/vector-icons/FontAwesome5';

const SCREEN_WIDTH = Dimensions.get('window').width;
const SWIPE_THRESHOLD = -80;
const DELETE_WIDTH = 80;

interface SwipeableRowProps {
  children: React.ReactNode;
  onDelete: () => void;
}

export function SwipeableRow({ children, onDelete }: SwipeableRowProps) {
  const translateX = useRef(new Animated.Value(0)).current;
  const isOpen = useRef(false);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => false,
      onMoveShouldSetPanResponder: (_, gesture) => {
        // Only activate for horizontal swipes
        return Math.abs(gesture.dx) > Math.abs(gesture.dy) && Math.abs(gesture.dx) > 5;
      },
      onPanResponderGrant: () => {
        // Stop any running animation
        translateX.stopAnimation();
      },
      onPanResponderMove: (_, gesture) => {
        const currentOffset = isOpen.current ? -DELETE_WIDTH : 0;
        const newValue = currentOffset + gesture.dx;
        
        // Clamp between -DELETE_WIDTH and 0
        if (newValue <= 0 && newValue >= -DELETE_WIDTH) {
          translateX.setValue(newValue);
        } else if (newValue > 0) {
          translateX.setValue(0);
        } else {
          translateX.setValue(-DELETE_WIDTH);
        }
      },
      onPanResponderRelease: (_, gesture) => {
        const currentOffset = isOpen.current ? -DELETE_WIDTH : 0;
        const finalPosition = currentOffset + gesture.dx;

        if (finalPosition < SWIPE_THRESHOLD) {
          // Open delete button
          Animated.spring(translateX, {
            toValue: -DELETE_WIDTH,
            useNativeDriver: true,
            friction: 8,
          }).start();
          isOpen.current = true;
        } else {
          // Close
          Animated.spring(translateX, {
            toValue: 0,
            useNativeDriver: true,
            friction: 8,
          }).start();
          isOpen.current = false;
        }
      },
    })
  ).current;

  const handleDelete = () => {
    // Animate out then delete
    Animated.timing(translateX, {
      toValue: -SCREEN_WIDTH,
      duration: 200,
      useNativeDriver: true,
    }).start(() => {
      onDelete();
    });
  };

  const closeRow = () => {
    Animated.spring(translateX, {
      toValue: 0,
      useNativeDriver: true,
      friction: 8,
    }).start();
    isOpen.current = false;
  };

  return (
    <View style={styles.container}>
      {/* Delete Button Background */}
      <View style={styles.deleteContainer}>
        <Pressable onPress={handleDelete} style={styles.deleteButton}>
          <FontAwesome5 name="trash-alt" size={18} color="#ffffff" />
          <Text style={styles.deleteText}>Delete</Text>
        </Pressable>
      </View>

      {/* Swipeable Content */}
      <Animated.View
        style={[
          styles.content,
          { transform: [{ translateX }] },
        ]}
        {...panResponder.panHandlers}
      >
        <Pressable onPress={closeRow}>
          {children}
        </Pressable>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: 12,
    borderRadius: 8,
    overflow: 'hidden',
  },
  deleteContainer: {
    position: 'absolute',
    right: 0,
    top: 0,
    bottom: 0,
    width: DELETE_WIDTH,
    backgroundColor: '#ef4444',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 8,
  },
  deleteButton: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 8,
  },
  deleteText: {
    color: '#ffffff',
    fontSize: 12,
    marginTop: 4,
    fontWeight: '500',
  },
  content: {
    backgroundColor: 'white',
    borderRadius: 8,
  },
});