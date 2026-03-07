/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "Inter", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "Space Grotesk", "system-ui", "sans-serif"],
      },
      colors: {
        accent: "#fbbf24",
        accentMuted: "#fcd34d",
        surface: "rgba(255,255,255,0.03)",
        border: "rgba(255,255,255,0.08)",
      },
    },
  },
  plugins: [],
};
