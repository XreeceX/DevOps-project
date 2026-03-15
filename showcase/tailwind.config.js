/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["system-ui", "sans-serif"],
        display: ["system-ui", "sans-serif"],
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
}
