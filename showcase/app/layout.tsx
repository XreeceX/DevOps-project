import type { Metadata } from "next";
import { Inter, Space_Grotesk } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-display",
});

export const metadata: Metadata = {
  title: "Fraud Detection | Reece Rodrigues",
  description: "ML-powered fraud detection for banking transactions. SMOTE + RandomForest—containerized for reproducible deployment.",
  openGraph: {
    title: "Fraud Detection | Reece Rodrigues",
    description: "ML-powered fraud detection for banking transactions. SMOTE + RandomForest—containerized for reproducible deployment.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${spaceGrotesk.variable} scroll-smooth`}>
      <body className="min-h-screen bg-[#0a0a0f] font-sans text-zinc-300 antialiased">
        {children}
      </body>
    </html>
  );
}
