'use client';

import React, { useState, useEffect } from 'react';
import { message } from 'antd';
import { apiRequest } from "../api/api";
import { API_ENDPOINTS } from "../api/endpoints";
import ModelSelector from './ModelSelector';
import ImageUploader from './ImageUploader';
import AnalysisResults from './AnalysisResults';
import Title from './Title';

export default function MainContent() {
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (file: File) => {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
        if (!file || !selectedModel) {
            throw new Error("Please select an image and AI model");
        }

        if (!file.type.startsWith("image/")) {
            throw new Error("Invalid file type. Please upload an image.");
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("model_name", selectedModel);
        console.log(formData.get("model_name"));
        const data = await apiRequest(API_ENDPOINTS.PREDICT, "POST", formData);
        
        // 创建图片的临时 URL
        const imageUrl = URL.createObjectURL(file);
        setResult({ ...data, image_url: imageUrl });
        
        message.success("Analysis completed successfully!");
    } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "An unexpected error occurred.";
        setError(errorMessage);
        message.error(errorMessage);
    } finally {
        setLoading(false);
    }
  };

  // 清理临时 URL
  useEffect(() => {
    return () => {
      if (result?.image_url) {
        URL.revokeObjectURL(result.image_url);
      }
    };
  }, [result?.image_url]);

  return (
    <div className="py-8">
      <div className="container mx-auto px-4">
        <div className="max-w-7xl mx-auto">
          <Title />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
            {/* Left Column */}
            <section className="bg-white rounded-lg shadow-sm border border-gray-100 p-6 shadow-xl">
              <div className="space-y-6">
                <ModelSelector
                  selectedModel={selectedModel}
                  onModelChange={setSelectedModel}
                />
                <ImageUploader
                  selectedModel={selectedModel}
                  onAnalyze={handleAnalyze}
                />
              </div>
            </section>

            {/* Right Column */}
            <section className="bg-white rounded-lg shadow-sm border border-gray-100 shadow-xl">
              <AnalysisResults
                loading={loading}
                result={result}
                error={error}
              />
            </section>
          </div>
        </div>
      </div>
    </div>
  );
} 