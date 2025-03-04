import { Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

interface ModelOption {
  value: string; // The actual value to be sent to API
  label: string; // The display name (previously description)
}

const modelOptions: ModelOption[] = [
  { value: "AlexNet", label: "Deeper architecture, good for general image classification" },
  { value: "DenseNet", label: "Dense connectivity pattern for improved feature reuse" },
  { value: "EfficientNet", label: "Balanced network scaling for better efficiency" },
  { value: "GoogLeNet", label: "Inception architecture, efficient and accurate" },
  { value: "LeNet", label: "Classic CNN architecture, lightweight and fast" },
  { value: "MobileNetV2", label: "Lightweight model optimized for mobile devices" },
  { value: "MobileNetV3", label: "Advanced mobile-optimized architecture" },
  { value: "ResNet", label: "Deep residual learning framework" },
  { value: "RegNet", label: "Systematically designed network architecture" },
  { value: "ShuffleNetV2", label: "Lightweight and efficient for mobile devices" },
  { value: "VGG", label: "Deep CNN with simple, uniform architecture" },
  { value: "VGGNet", label: "VGG Net architecture" }
];

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (value: string) => void;
}

export default function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  const selectedModelInfo = modelOptions.find((model) => model.value === selectedModel);

  const handleChange = (value: string | null) => {
    onModelChange(value || "");
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-base font-medium text-gray-900">Select AI Model</h3>
          <Tooltip title="Choose a model for waste classification">
            <InfoCircleOutlined className="text-gray-400 cursor-help" />
          </Tooltip>
        </div>
        {selectedModel && (
          <Tooltip title={selectedModelInfo?.label}>
            <span className="text-sm text-emerald-600 cursor-help">
              Using: {selectedModel}
            </span>
          </Tooltip>
        )}
      </div>

      <Select
        className="w-full"
        placeholder="Select an AI model"
        options={modelOptions.map((model) => ({
          value: model.value,
          label: (
            <div>
              <div className="font-medium">{model.value}</div>
              <div className="text-xs text-gray-500">{model.label}</div>
            </div>
          ),
        }))}
        labelInValue
        value={selectedModel ? { value: selectedModel, label: selectedModel } : undefined}
        onChange={(selected) => handleChange(selected?.value)}
        showSearch
        filterOption={(input, option) =>
          (option?.value?.toString().toLowerCase() || "").includes(input.toLowerCase()) ||
          (option?.label?.toString().toLowerCase() || "").includes(input.toLowerCase())
        }
        allowClear
      />
      {selectedModelInfo && (
        <p className="mt-2 text-sm text-gray-500">
          {selectedModelInfo.label}
        </p>
      )}
    </div>
  );
}
