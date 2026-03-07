import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
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
    <html lang="en" className={`${inter.variable} scroll-smooth`}>
      <body className="min-h-screen bg-zinc-950 font-sans antialiased text-zinc-100">
        {children}
      </body>
    </html>
  );
}
