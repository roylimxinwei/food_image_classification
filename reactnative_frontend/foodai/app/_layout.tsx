import "../global.css";
import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { Stack } from "expo-router";
import { useColorScheme } from '@/hooks/use-color-scheme';
import { GluestackUIProvider } from '@/components/ui/gluestack-ui-provider';

export default function RootLayout() {
  const colorScheme = useColorScheme();

  return (
    <GluestackUIProvider mode="light">
      <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
          <Stack>
            <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
            <Stack.Screen name="analysis" options={{ headerShown: false }} />
            <Stack.Screen name="nutrition" options={{ headerShown: false }} />
            <Stack.Screen name="segmentation" options={{ headerShown: false }} />
            <Stack.Screen name="summary" options={{ headerShown: false }} />
          </Stack>
      </ThemeProvider>
    </GluestackUIProvider>
  );
}
