import { FaRecycle, FaGithub, FaRobot, FaUsers } from 'react-icons/fa';
import Link from 'next/link';

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-16">
            <div className="flex justify-center mb-6">
              <FaRecycle className="text-emerald-600 text-6xl" />
            </div>
            <h1 className="text-4xl font-bold text-gray-900 mb-4">About WasteClassifier AI</h1>
            <p className="text-xl text-gray-600">
              Empowering sustainable waste management through artificial intelligence
            </p>
          </div>

          {/* Mission Section */}
          <div className="bg-white rounded-lg shadow-sm p-8 mb-8">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Our Mission</h2>
            <p className="text-gray-600 mb-4">
              WasteClassifier AI aims to revolutionize waste management by providing instant, accurate waste classification using advanced AI technology. Our goal is to help individuals and organizations make better decisions about waste disposal, contributing to a more sustainable future.
            </p>
          </div>

          {/* Features Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <div className="flex items-center gap-3 mb-4">
                <FaRobot className="text-emerald-600 text-2xl" />
                <h3 className="text-xl font-semibold text-gray-900">AI-Powered Classification</h3>
              </div>
              <p className="text-gray-600">
                Utilizing state-of-the-art deep learning models to accurately identify and classify different types of waste materials.
              </p>
            </div>
            <div className="bg-white rounded-lg shadow-sm p-6">
              <div className="flex items-center gap-3 mb-4">
                <FaUsers className="text-emerald-600 text-2xl" />
                <h3 className="text-xl font-semibold text-gray-900">User-Friendly Interface</h3>
              </div>
              <p className="text-gray-600">
                Simple and intuitive interface that makes waste classification accessible to everyone.
              </p>
            </div>
          </div>

          {/* Technology Stack */}
          <div className="bg-white rounded-lg shadow-sm p-8 mb-8">
            <h2 className="text-2xl font-semibold text-gray-900 mb-6">Technology Stack</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="font-medium text-gray-900">Next.js</p>
                <p className="text-sm text-gray-600">Frontend Framework</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="font-medium text-gray-900">Python</p>
                <p className="text-sm text-gray-600">Backend</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="font-medium text-gray-900">PyTorch</p>
                <p className="text-sm text-gray-600">AI Framework</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="font-medium text-gray-900">Docker</p>
                <p className="text-sm text-gray-600">Containerization</p>
              </div>
            </div>
          </div>

          {/* GitHub Section */}
          <div className="bg-white rounded-lg shadow-sm p-8">
            <div className="flex items-center gap-3 mb-4">
              <FaGithub className="text-gray-900 text-2xl" />
              <h2 className="text-2xl font-semibold text-gray-900">Open Source</h2>
            </div>
            <p className="text-gray-600 mb-6">
              WasteClassifier AI is an open-source project. We welcome contributions from the community to help improve waste classification and make it more accessible to everyone.
            </p>
            <Link 
              href="https://github.com/AlanY1an/Smartbin" 
              target="_blank"
              className="inline-flex items-center gap-2 px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors"
            >
              <FaGithub />
              View on GitHub
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
} 