import Link from 'next/link';
import { FaRecycle } from "react-icons/fa";
import { useState } from 'react';
import { MenuOutlined, CloseOutlined } from '@ant-design/icons';

export default function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <header className="border-b border-gray-200 bg-white shadow-2xs sticky top-0 z-10">
      <div className="container mx-auto px-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center h-16">
          <div className="flex items-center gap-2">
            <FaRecycle className="text-emerald-600 text-xl" />
            <span className="text-xl font-semibold">WasteClassifier AI</span>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            <Link href="/documentation" className="text-gray-600 hover:text-gray-900">
              Documentation
            </Link>
            <Link href="/about" className="text-gray-600 hover:text-gray-900">
              About
            </Link>
            <button className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors">
              Get Started
            </button>
          </nav>

          {/* Mobile Menu Button */}
          <button 
            className="md:hidden p-2 hover:bg-gray-100 rounded-lg transition-colors"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? <CloseOutlined className="text-xl" /> : <MenuOutlined className="text-xl" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        <div 
          className={`md:hidden border-t border-gray-200 overflow-hidden transition-all duration-300 ease-in-out ${
            isMenuOpen ? 'max-h-64 opacity-100' : 'max-h-0 opacity-0'
          }`}
        >
          <nav className="flex flex-col py-4 space-y-4">
            <Link 
              href="/documentation" 
              className="text-gray-600 hover:text-gray-900 px-2 py-2 hover:bg-gray-50 rounded-lg transition-colors duration-200"
              onClick={() => setIsMenuOpen(false)}
            >
              Documentation
            </Link>
            <Link 
              href="/about" 
              className="text-gray-600 hover:text-gray-900 px-2 py-2 hover:bg-gray-50 rounded-lg transition-colors duration-200"
              onClick={() => setIsMenuOpen(false)}
            >
              About
            </Link>
            <button 
              className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors duration-200 mx-2"
              onClick={() => setIsMenuOpen(false)}
            >
              Get Started
            </button>
          </nav>
        </div>
      </div>
    </header>
  );
} 