import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
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
    <html lang="en" className={`${spaceGrotesk.variable} scroll-smooth`}>
      <body className="min-h-screen bg-[#0c1222] font-sans antialiased text-zinc-100">
        {children}
      </body>
    </html>
  );
}
