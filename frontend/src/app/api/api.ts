import { API_BASE_URL, API_ENDPOINTS } from "./endpoints";

export const apiRequest = async (endpoint: string, method: string, body: FormData | null = null) => {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, { method, body });
        console.log("API Base URL:", API_BASE_URL);
        console.log("Full API URL:", `${API_BASE_URL}${API_ENDPOINTS.PREDICT}`);
        
        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error("Invalid response from server.");
        }

        if (!response.ok) {
            throw new Error(data.error || "Something went wrong.");
        }

        return data;
    } catch (error) {
        console.error("API Error:", error);
        throw error;
    }
};
