import { Pressable, Text } from "react-native";
import { forwardRef } from "react";

interface ButtonProps {
  children: React.ReactNode;
  variant?: "solid" | "outline" | "ghost";
  size?: "sm" | "md" | "lg";
  action?: "primary" | "secondary" | "positive" | "negative";
  onPress?: () => void;
  className?: string;
  disabled?: boolean;
}

interface ButtonTextProps {
  children: React.ReactNode;
  className?: string;
}

const actionColors = {
  primary: {
    solid: "bg-blue-500 active:bg-blue-600",
    outline: "border-2 border-blue-500",
    ghost: "bg-transparent",
    text: "text-blue-500",
  },
  secondary: {
    solid: "bg-gray-500 active:bg-gray-600",
    outline: "border-2 border-gray-500",
    ghost: "bg-transparent",
    text: "text-gray-500",
  },
  positive: {
    solid: "bg-green-500 active:bg-green-600",
    outline: "border-2 border-green-500",
    ghost: "bg-transparent",
    text: "text-green-500",
  },
  negative: {
    solid: "bg-red-500 active:bg-red-600",
    outline: "border-2 border-red-500",
    ghost: "bg-transparent",
    text: "text-red-500",
  },
};

const sizeClasses = {
  sm: "px-3 py-1.5",
  md: "px-4 py-2",
  lg: "px-6 py-3",
};

export const Button = forwardRef<any, ButtonProps>(
  (
    {
      children,
      variant = "solid",
      size = "md",
      action = "primary",
      onPress,
      className = "",
      disabled = false,
    },
    ref
  ) => {
    const colorClass = actionColors[action][variant];
    
    return (
      <Pressable
        ref={ref}
        onPress={onPress}
        disabled={disabled}
        className={`rounded-lg items-center justify-center ${colorClass} ${sizeClasses[size]} ${disabled ? "opacity-50" : ""} ${className}`}
      >
        {children}
      </Pressable>
    );
  }
);

export const ButtonText = ({ children, className = "" }: ButtonTextProps) => {
  return (
    <Text className={`font-semibold text-white ${className}`}>{children}</Text>
  );
};

Button.displayName = "Button";
