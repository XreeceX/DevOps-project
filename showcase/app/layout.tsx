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
      {/* ensure correct theme before React hydrates */}
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('theme');if(t==='dark')document.documentElement.classList.add('dark');}catch(e){}})()`
          }}
        />
      </head>
      <body className="min-h-screen bg-white dark:bg-[#0a0a0f] font-sans text-zinc-900 dark:text-zinc-300 antialiased">
        {children}
      </body>
    </html>
  );
}
