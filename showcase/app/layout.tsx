import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Fraud Detection | DevOps Project",
  description: "ML-powered fraud detection comparing SMOTE vs baseline RandomForest for imbalanced banking transaction data.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.className} min-h-screen bg-ink text-[#e2e8f0] antialiased`}>
        {children}
      </body>
    </html>
  );
}
