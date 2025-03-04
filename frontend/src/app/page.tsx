'use client';

import Header from './components/Header';
import Footer from './components/Footer';
import Title from './components/Title';
import MainContent from './components/MainContent';
import { Analytics } from "@vercel/analytics/react";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Header />
      <div className="flex-1 py-8 space-y-8">
        <Title />
        <MainContent />
      </div>
      <Footer />
      <Analytics />
    </div>
  );
}
