// API Configuration
// Update this IP when switching networks

// To find your Windows IP, run in PowerShell: ipconfig | findstr "IPv4"
// Use the IP address that matches your current network (WiFi or Hotspot)

export const API_CONFIG = {
    // Change this IP when switching networks
    // BASE_URL: "http://192.168.0.120:8000",
    BASE_URL: "https://superrighteous-prepositionally-tajuana.ngrok-free.dev",
    
    // API endpoints
    ENDPOINTS: {
        ROOT: "/",
        HEALTH: "/health",
        SEGMENT: "/segment",
    }
};

// Helper function to build full URL
export const getApiUrl = (endpoint: keyof typeof API_CONFIG.ENDPOINTS): string => {
    return `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS[endpoint]}`;
};

export const API_V1 = `${API_CONFIG.BASE_URL}/api/v1`;
