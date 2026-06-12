import { Text, View } from "react-native";
import {useState, useEffect} from 'react';
import { getApiUrl } from '@/config/api';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function SettingsTab() {
      const [data, setData] = useState(null);
      const [loading, setLoading] = useState(true);
      const [error, setError] = useState(null);
  
      useEffect(() => {
          const fetchData = async () => {
              try {
                  const response = await fetch(getApiUrl('HEALTH'));
                  if (!response.ok) {
                      throw new Error('error fetching data');
                  }
                  const json = await response.json();
                  setData(json);
              }
              catch (err: any) {
                  setError(err.message);
              } finally {
                  setLoading(false);
              }
          };
          fetchData();
      }, []);
  
      if (loading) {
          return <SafeAreaView><Text>Loading...</Text></SafeAreaView>;
      }
      if (error) {
          return <SafeAreaView><Text>Error: {error}</Text></SafeAreaView>;
      }
  return (
    <View
      style={{
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Text>Settings Page.</Text>
      <Text>{data ? JSON.stringify(data) : 'No data'}</Text>
    </View>
  );
}