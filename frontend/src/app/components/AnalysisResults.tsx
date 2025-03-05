import { Card, Spin, Image } from 'antd';
import { FileSearchOutlined, WarningOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { BsRecycle } from 'react-icons/bs';
import { IoDocumentTextOutline } from 'react-icons/io5';

interface AnalysisResult {
  model: string;
  prediction: string;
  confidence: string;
  total_time: string;
  image_url?: string;
  waste_category: string;
  disposal_message: string;
  disposal_guidelines: string;
}

interface AnalysisResultsProps {
  loading: boolean;
  result: AnalysisResult | null;
  error: string | null;
}

const ConfidenceBar = ({ value }: { value: number }) => {
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm text-gray-600">Confidence Level</span>
        <span className="text-sm font-medium text-emerald-600">{value}%</span>
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2.5">
        <div 
          className="bg-emerald-600 h-2.5 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
};

export default function AnalysisResults({ loading, result, error }: AnalysisResultsProps) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center p-6">
        <Spin size="large" />
        <p className="mt-4 text-gray-600 text-sm">Analyzing your image...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center p-6">
        <div className="text-red-500 mb-4">
          <WarningOutlined style={{ fontSize: '48px' }} />
        </div>
        <h3 className="text-lg font-medium text-red-600">Analysis Error</h3>
        <p className="mt-2 text-center text-sm text-red-500">
          {error}
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center p-6 text-gray-400">
        <FileSearchOutlined style={{ fontSize: '48px' }} />
        <h3 className="mt-4 text-lg font-medium text-gray-600">No Analysis Results Yet</h3>
        <p className="mt-2 text-center text-sm text-gray-500">
          Upload an image and click analyze to see the results here
        </p>
      </div>
    );
  }

  const { prediction, confidence, total_time, model, image_url, waste_category, disposal_message, disposal_guidelines } = result;
  const predictionLower = prediction.toLowerCase();
  const confidenceValue = parseFloat(confidence);

  return (
    <div className="p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4 border-b border-gray-200 pb-3">Analysis Results</h2>
      
      <div className="flex flex-col md:flex-row justify-between gap-6">
        <div className="w-full md:w-[200px] flex-shrink-0 mb-4 md:mb-0">
          <div className="aspect-square rounded-lg overflow-hidden border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
            <img
              src={image_url || '/placeholder.png'}
              alt="Analyzed waste item"
              className="w-full h-full object-cover"
            />
          </div>
        </div>

        <div className="flex-1">
          <div className="text-3xl font-bold text-gray-900 mb-4">
            {prediction}
          </div>
          
          <div className="mb-6">
            <ConfidenceBar value={confidenceValue} />
          </div>

          <div className="bg-emerald-50 rounded-lg p-4 border border-emerald-100 mb-2">
            <h3 className="flex items-center gap-2 text-sm font-medium text-gray-900">
              <BsRecycle className="text-emerald-600 text-lg" />
              <span>Waste Category</span>
            </h3>
            <div className="mt-3 flex items-center gap-2">
              <span className="text-lg font-medium text-emerald-700">{waste_category}</span>
              <span className="px-2 py-1 bg-emerald-100 text-emerald-700 text-xs rounded-full">
                Recommended
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-emerald-50 rounded-lg p-4 border border-emerald-100 mb-2">
          <h3 className="flex items-center gap-2 text-sm font-medium text-gray-900">
            <InfoCircleOutlined className="text-emerald-600 text-lg" />
            <span>Disposal Instructions</span>
          </h3>
          <p className="mt-2 text-sm text-gray-600 pl-7">
            {disposal_message}
          </p>
        </div>

        <div className="bg-blue-50 rounded-lg p-4 border border-emerald-100 mb-1">
          <h3 className="flex items-center gap-2 text-sm font-medium text-gray-900">
            <IoDocumentTextOutline className="text-emerald-600 text-lg" />
            <span>Additional Guidelines</span>
          </h3>
          <p className="mt-2 text-sm text-gray-600 pl-7">
            {disposal_guidelines}
          </p>
        </div>
      </div>

      <div className="text-right text-xs text-gray-500 mt-0">
        Processing Time: {total_time}
      </div>
    </div>
  );
} 