import { useState } from 'react';
import { Upload, Button, message } from 'antd';
import { CloudUploadOutlined, InboxOutlined, SearchOutlined } from '@ant-design/icons';
import type { UploadProps, UploadFile } from 'antd';

const { Dragger } = Upload;

interface ImageUploaderProps {
  selectedModel: string;
  onAnalyze: (file: File) => Promise<void>;
}

export default function ImageUploader({
  selectedModel,
  onAnalyze,
}: ImageUploaderProps) {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [analyzing, setAnalyzing] = useState(false);

  const uploadProps: UploadProps = {
    name: 'file',
    multiple: false,
    maxCount: 1,
    fileList,
    accept: 'image/*',
    beforeUpload: (file) => {
      setFileList([file]);
      return false; // Prevent auto upload
    },
    onRemove: () => {
      setFileList([]);
    },
  };

  const handleAnalyzeClick = async () => {
    if (!selectedModel) {
      message.error('Please select a model first');
      return;
    }

    if (!fileList.length) {
      message.error('Please upload an image first');
      return;
    }

    try {
      setAnalyzing(true);
      await onAnalyze(fileList[0] as unknown as File);
      message.success('Analysis completed successfully!');
    } catch (error) {
      message.error('Analysis failed. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-base font-medium text-gray-900 mb-4">Upload Image</h3>
        <Dragger {...uploadProps}>
          <p className="ant-upload-drag-icon !mb-px">
            <CloudUploadOutlined className="!text-gray-400" />
          </p>
          <p className="ant-upload-text">
            Drag and drop your image here
          </p>
          <p className="ant-upload-hint">
            or click to browse files
          </p>
        </Dragger>
      </div>

      <Button
        type="primary"
        icon={<SearchOutlined />}
        size="large"
        block
        loading={analyzing}
        disabled={!fileList.length || !selectedModel || analyzing}
        onClick={handleAnalyzeClick}
        className="!text-white !bg-emerald-600 hover:!bg-emerald-500"
      >
        {analyzing ? 'Analyzing...' : 'Analyze Image'}
      </Button>

      {fileList.length > 0 && !selectedModel && (
        <p className="text-amber-600 text-sm">
          Please select a model to analyze your image
        </p>
      )}
    </div>
  );
}
