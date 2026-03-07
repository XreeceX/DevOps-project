import type { Metadata } from "next";
import { Inter, Syne } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
});

const syne = Syne({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-display",
});

export const metadata: Metadata = {
  title: "Fraud Detection | DevOps Project",
  description: "ML-powered fraud detection comparing SMOTE vs baseline RandomForest for imbalanced banking transaction data.",
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
    <html lang="en" className={`${inter.variable} ${syne.variable} scroll-smooth`}>
      <body className="min-h-screen bg-ink font-sans text-slate-200 antialiased">
        {children}
      </body>
    </html>
  );
}
