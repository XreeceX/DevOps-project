import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
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
    <html lang="en" className={`${outfit.variable} scroll-smooth`}>
      <body className="min-h-screen bg-[#fafbfc] font-sans antialiased text-zinc-800">
        {children}
      </body>
    </html>
  );
}
