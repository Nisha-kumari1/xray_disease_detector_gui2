import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AuraRad | Premium AI X-Ray Analysis",
  description: "Industry-level AI-powered medical imaging system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-slate-50 min-h-screen text-slate-900`}>
        {/* Natural Professional Background */}
        <div className="fixed inset-0 -z-10 bg-[#f8fafc]">
          {/* Subtle clinical blue gradient at the top */}
          <div className="absolute top-0 w-full h-[500px] bg-gradient-to-b from-blue-50/80 to-transparent pointer-events-none"></div>
        </div>
        
        <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-6 h-screen flex flex-col">
          {children}
        </main>
      </body>
    </html>
  );
}
